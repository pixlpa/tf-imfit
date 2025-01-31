from __future__ import print_function
import re, os, sys, argparse
from datetime import datetime
from collections import namedtuple
import tensorflow as tf
import numpy as np
from PIL import Image
import traceback
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
try:
    import imageio
except ImportError:
    print("Warning: imageio not installed. Image saving/loading may fail.")
try:
    import cv2
except ImportError:
    print("Warning: opencv-python not installed. Snapshot labels will be disabled.")

COLORMAP = None

InputsTuple = namedtuple('InputsTuple',
                         'input_image, weight_image, '
                         'x, y, target_tensor, max_row')

ModelsTuple = namedtuple('ModelsTuple',
                         'full, local, preview')

StateTuple = namedtuple('StateTuple', 'params, gabor, con_loss')

######################################################################
# Indexes of Gabor function parameters

GABOR_PARAM_U  = 0  # [-1, 1]
GABOR_PARAM_V  = 1  # [-1, 1]
GABOR_PARAM_R  = 2  # [0, 2*pi]
GABOR_PARAM_P0 = 3  # [0, 2*pi]
GABOR_PARAM_P1 = 4  # [0, 2*pi]
GABOR_PARAM_P2 = 5  # [0, 2*pi]
GABOR_PARAM_L  = 6  # [2.5*px, 4]
GABOR_PARAM_T  = 7  # [px, 4]
GABOR_PARAM_S  = 8  # [px, 2]
GABOR_PARAM_H0 = 9  # [0, 2]
GABOR_PARAM_H1 = 10 # [0, 2]
GABOR_PARAM_H2 = 11 # [0, 2]

GABOR_NUM_PARAMS = 12

GABOR_RANGE = np.array([
    [ -1, 1 ],
    [ -1, 1 ],
    [ -np.pi, np.pi ],
    [ -np.pi, np.pi ],
    [ -np.pi, np.pi ],
    [ -np.pi, np.pi ],
    [ 0, 4 ],
    [ 0, 4 ],
    [ 0, 2 ],
    [ 0, 2 ],
    [ 0, 2 ],
    [ 0, 2 ] ])

######################################################################
# Parse a duration string

def parse_duration(dstr):

    expr = r'^(([0-9]+)(:|h))?([0-9]+)((:|m)(([0-9]+)s?))?$'

    g = re.match(expr, dstr)

    if g is None:
        raise argparse.ArgumentTypeError(dstr + ': invalid duration format')

    def make_int(x):
        if x is None:
            return 0
        else:
            return int(x)

    h = make_int( g.group(2) )
    m = make_int( g.group(4) )
    s = make_int( g.group(8) )

    return (h*60 + m)*60 + s


######################################################################
# Parse command-line options, return namespace containing results

def get_options():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('image', type=argparse.FileType('rb'),
                        metavar='IMAGE.png',
                        help='image to approximate')
    
    parser.add_argument('-s', '--max-size', type=int, metavar='N',
                        help='maximum size of image to load',
                        default=128)

    parser.add_argument('-p', '--preview-size', type=int, metavar='N',
                        default=0,
                        help='size of large preview image (0 to disable)')
    
    parser.add_argument('-w', '--weights', type=argparse.FileType('rb'),
                        metavar='WEIGHTS.png',
                        help='load weights from file',
                        default=None)

    parser.add_argument('-i', '--input', type=str,
                        metavar='PARAMFILE.txt',
                        help='read input params from file')

    parser.add_argument('-o', '--output', type=str,
                        metavar='PARAMFILE.txt',
                        help='write input params to file')
    
    parser.add_argument('-t', '--time-limit', type=parse_duration,
                        metavar='LIMIT',
                        help='time limit (e.g. 1:30 or 1h30m)',
                        default=None)

    parser.add_argument('-T', '--total-iterations', type=int,
                        metavar='N',
                        help='total limit on outer loop iterations',
                        default=None)

    parser.add_argument('-F', '--full-every', type=int, metavar='N',
                        default=32,
                        help='perform joint optimization '
                        'after every N outer loops')

    parser.add_argument('-f', '--full-iter', type=int, metavar='N',
                        help='maximum # of iterations for joint optimization',
                        default=10000)

    parser.add_argument('-r', '--local-learning-rate', type=float, metavar='R',
                        help='learning rate for local opt.',
                        default=0.01)

    parser.add_argument('-R', '--full-learning-rate', type=float, metavar='R',
                        help='learning rate for full opt.',
                        default=0.0005)
    
    parser.add_argument('-n', '--num-models', type=int, metavar='N',
                        help='total number of models to fit',
                        default=128)

    parser.add_argument('-L', '--num-local', type=int, metavar='N',
                        help='number of random guesses per local fit',
                        default=200)

    parser.add_argument('-l', '--local-iter', type=int, metavar='N',
                        help='maximum # of iterations per local fit',
                        default=100)
     
    parser.add_argument('-P', '--perturb-amount', type=float,
                        metavar='R', default=0.15,
                        help='amount to perturb replacement fits by')

    parser.add_argument('-c', '--copy-quantity', type=float,
                        metavar='C',
                        help='number or fraction of re-fits'
                        'to initialize with cur. model',
                        default=0.5)
    
    parser.add_argument('-a', '--anneal-temp', type=float, metavar='T',
                        help='temperature for simulated annealing',
                        default=0.08)

    parser.add_argument('-S', '--label-snapshot', action='store_true',
                        help='individually label snapshots')
 
    parser.add_argument('-x', '--snapshot-prefix', type=str,
                        metavar='BASENAME',
                        help='prefix for snapshots', default='out')
  
    opts = parser.parse_args()

    if opts.copy_quantity < 0:
        opts.copy_quantity = 0
    elif opts.copy_quantity >= 1:
        opts.copy_quantity = 1
    else:
        opts.copy_quantity = int(round(opts.copy_quantity * opts.num_local))

    if opts.preview_size < 0:
        opts.preview_size = 0

    return opts

######################################################################
# Compute x/y coordinates for a grid spanning [-1, 1] for the given
# image shape (h, w)

def normalized_grid(shape):

    h, w = shape
    hwmax = max(h, w)

    px = 2.0 / hwmax

    x = (np.arange(w, dtype=np.float32) - 0.5*(w) + 0.5) * px
    y = (np.arange(h, dtype=np.float32) - 0.5*(h) + 0.5) * px

    return px, x, y

######################################################################
# Proportional scaling for image shape

def scale_shape(shape, desired_size):

    h, w = shape

    if w > h:
        wnew = desired_size
        hnew = int(round(float(h) * desired_size / w))
    else:
        wnew = int(round(float(w) * desired_size / h))
        hnew = desired_size

    return hnew, wnew

######################################################################
# Open an image, convert it to grayscale, resize to desired size
        
def open_image(handle, max_size, grayscale):

    if handle is None:
        return None

    image = Image.open(handle)

    if grayscale:
        if image.mode != 'L':
            print('converting {} to grayscale'.format(handle.name))
            image = image.convert('L')
    else:
      if image.mode != 'RGB':
          image = image.convert('RGB')
      assert image.mode=='RGB'

    w, h = image.size
    
    if max(w, h) > max_size:
        h, w = scale_shape((h, w), max_size)
        image = image.resize((w, h), resample=Image.LANCZOS)

    print('{} is {}x{}'.format(handle.name, w, h))

    image = np.array(image).astype(np.float32) / 255.
    
    return image

######################################################################
# Set up all of the tensorflow inputs to our models

def setup_inputs(opts):

    input_image = open_image(opts.image, opts.max_size, grayscale=False)
    print('  {} {} {}'.format(opts.image.name, input_image.shape, input_image.dtype))

    if opts.weights is not None:
        weight_image = open_image(opts.weights, opts.max_size, grayscale=False)
        assert weight_image.size == input_image.size
    else:
        weight_image = 1.0

    # move to -1, 1 range for input image
    input_image = input_image * 2 - 1
    
    px, x, y = normalized_grid(input_image.shape[:2])

    GABOR_RANGE[GABOR_PARAM_L, 0] = 2.5*px
    GABOR_RANGE[GABOR_PARAM_T, 0] = px
    GABOR_RANGE[GABOR_PARAM_S, 0] = px
    
    target_tensor = tf.compat.v1.placeholder(tf.float32,
                                   shape=input_image.shape,
                                   name='target')

    max_row = tf.compat.v1.placeholder(tf.int32, shape=(),
                             name='max_row')

    return InputsTuple(input_image, weight_image,
                       x, y, target_tensor, max_row)

######################################################################
# Encapsulate the tensorflow objects we need to run our fit.
# Note we will create several of these (see main function below).

class GaborModel:
    def __init__(self, name, count, image_shape):
        self.name = name
        self.count = count
        self.image_shape = image_shape
        self.variables = {}

        # Create variables with proper bounds
        h, w = image_shape[:2]
        self.variables.update({
            # Position variables bounded by image dimensions
            'pos_x': tf.Variable(tf.random.uniform([count], 0, w), 
                               name=f'{name}/pos_x'),
            'pos_y': tf.Variable(tf.random.uniform([count], 0, h), 
                               name=f'{name}/pos_y'),
            # Sigma (size) with reasonable bounds
            'sigma': tf.Variable(tf.random.uniform([count], 1, 20), 
                               name=f'{name}/sigma'),
            # Theta (rotation) between 0 and pi
            'theta': tf.Variable(tf.random.uniform([count], 0, np.pi), 
                               name=f'{name}/theta'),
            # Frequency with reasonable bounds
            'frequency': tf.Variable(tf.random.uniform([count], 0.02, 0.5), 
                                   name=f'{name}/frequency'),
            # Phase between 0 and 2pi
            'phase': tf.Variable(tf.random.uniform([count], 0, 2*np.pi), 
                               name=f'{name}/phase'),
            # Colors between 0 and 1
            'colors': tf.Variable(tf.random.uniform([count, 3], 0, 1), 
                                name=f'{name}/colors'),
            # Amplitude between 0 and 1
            'amplitude': tf.Variable(tf.random.uniform([count], 0, 1), 
                                   name=f'{name}/amplitude'),
            'scale': tf.Variable(tf.ones([count]), name=f'{name}/scale')
        })

        # Property accessors
        for name in self.variables:
            setattr(self, name, self.variables[name])

        # Allow evaluating less than ensemble_size models (i.e. while
        # building up full model).
        if max_row is None:
            max_row = ensemble_size

        gmin = GABOR_RANGE[:,0].reshape(1,GABOR_NUM_PARAMS,1).copy()
        gmax = GABOR_RANGE[:,1].reshape(1,GABOR_NUM_PARAMS,1).copy()
            
        # Parameter tensor could be passed in or created here
        if params is not None:

            self.params = params

        else:

            if initializer is None:
                initializer = tf.random_uniform_initializer(minval=gmin,
                                                            maxval=gmax,
                                                            dtype=tf.float32)

            # n x 12 x e
            self.params = tf.Variable(
                tf.random.uniform([num_parallel, GABOR_NUM_PARAMS, ensemble_size],
                                  minval=gmin, maxval=gmax, dtype=tf.float32),
                name='params')

        gmin[:,:GABOR_PARAM_L,:] = -np.inf
        gmax[:,:GABOR_PARAM_L,:] =  np.inf

        # n x 12 x e
        self.cparams = tf.clip_by_value(self.params[:,:,:max_row],
                                        gmin, gmax,
                                        name='cparams')


        ############################################################
        # Now compute the Gabor function for each fit/model

        # n x h x w x c x e
        
        # n x 1 x 1 x 1 x e
        u = self.cparams[:,None,None,None,GABOR_PARAM_U,:]
        v = self.cparams[:,None,None,None,GABOR_PARAM_V,:]
        r = self.cparams[:,None,None,None,GABOR_PARAM_R,:]
        l = self.cparams[:,None,None,None,GABOR_PARAM_L,:]
        t = self.cparams[:,None,None,None,GABOR_PARAM_T,:]
        s = self.cparams[:,None,None,None,GABOR_PARAM_S,:]
        
        p = self.cparams[:,None,None,GABOR_PARAM_P0:GABOR_PARAM_P0+3,:]
        h = self.cparams[:,None,None,GABOR_PARAM_H0:GABOR_PARAM_H0+3,:]


        cr = tf.cos(r)
        sr = tf.sin(r)

        f = np.float32(2*np.pi) / l

        s2 = s*s
        t2 = t*t

        # n x 1 x w x 1 x e
        xp = x-u

        # n x h x 1 x 1 x e
        yp = y-v

        # n x h x w x 1 x e
        b1 =  cr*xp + sr*yp
        b2 = -sr*xp + cr*yp

        b12 = b1*b1
        b22 = b2*b2

        w = tf.exp(-b12/(2*s2) - b22/(2*t2))

        k = f*b1 +  p
        ck = tf.cos(k)

        # n x h x w x c x e
        self.gabor = tf.identity(h * w * ck, name='gabor')

        ############################################################
        # Compute the ensemble sum of all models for each fit        
        
        # n x h x w x c
        self.approx = tf.reduce_sum(self.gabor, axis=4, name='approx')

        ############################################################
        # Everything below here is for optimizing, if we just want
        # to visualize, stop now.
        
        if target is None:
            return

        ############################################################
        # Compute loss for soft constraints
        #
        # All constraint losses are of the form min(c, 0)**2, where c
        # is an individual constraint function. So we only get a
        # penalty if the constraint function c is less than zero.
        
        # Pair-wise constraints on l, s, t:

        # n x e 
        l = self.cparams[:,GABOR_PARAM_L,:]
        s = self.cparams[:,GABOR_PARAM_S,:]
        t = self.cparams[:,GABOR_PARAM_T,:]

        pairwise_constraints = [
            s - l/32,
            l/2 - s,
            t - s,
            8*s - t
        ]
                
        # n x e x k
        self.constraints = tf.stack( pairwise_constraints,
                                     axis=2, name='constraints' )

        # n x e x k
        con_sqr = tf.minimum(self.constraints, 0)**2

        # n x e
        self.con_losses = tf.reduce_sum(con_sqr, axis=2, name='con_losses')

        # n (sum across mini-batch)
        self.con_loss_per_fit = tf.reduce_sum(self.con_losses, axis=1,
                                              name='con_loss_per_fit')



        ############################################################
        # Compute loss for approximation error
        
        # n x h x w x c
        self.err = tf.multiply((target - self.approx),
                                weight, name='err')

        err_sqr = 0.5*self.err**2

        # n (average across h/w/c)
        self.err_loss_per_fit = tf.reduce_mean(err_sqr, axis=(1,2,3),
                                               name='err_loss_per_fit')


        ############################################################
        # Compute various sums/means of above losses:

        # n
        self.loss_per_fit = tf.add(self.con_loss_per_fit,
                                   self.err_loss_per_fit,
                                   name='loss_per_fit')

        # scalars
        self.err_loss = tf.reduce_mean(self.err_loss_per_fit, name='err_loss')
        self.con_loss = tf.reduce_mean(self.con_loss_per_fit, name='con_loss')
        self.loss = self.err_loss + self.con_loss

        with tf.compat.v1.variable_scope('imfit_optimizer'):
            self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.opt.minimize(self.loss)

######################################################################
# Set up tensorflow models themselves. We need a separate model for
# each combination of inputs/dimensions to optimize.

def setup_models(opts, inputs):

    weight_tensor = tf.constant(inputs.weight_image)

    x_tensor = tf.constant(inputs.x.reshape(1,1,-1,1,1))
    y_tensor = tf.constant(inputs.y.reshape(1,-1,1,1,1))

    with tf.compat.v1.variable_scope('full'):

        full = GaborModel(1, opts.num_models,
                          x_tensor, y_tensor,
                          weight_tensor, inputs.target_tensor,
                          learning_rate=opts.full_learning_rate,
                          max_row = inputs.max_row,
                          initializer=tf.zeros_initializer())
    
    with tf.compat.v1.variable_scope('local'):
        
        local = GaborModel(opts.num_local, 1,
                           x_tensor, y_tensor,
                           weight_tensor, inputs.target_tensor,
                           learning_rate=opts.local_learning_rate)
        

    if opts.preview_size:

        preview_shape = scale_shape(map(int, inputs.target_tensor.shape[:2]),
                                    opts.preview_size)

        _, x_preview, y_preview = normalized_grid(preview_shape[:2])
        
        with tf.compat.v1.variable_scope('preview'):
            preview = GaborModel(1, opts.num_models,
                                 x_preview.reshape(1,1,-1,1,1),
                                 y_preview.reshape(1,-1,1,1,1),
                                 weight_tensor, target=None,
                                 max_row=inputs.max_row,
                                 params=full.params)

    else:

        preview = None

    return ModelsTuple(full, local, preview)

######################################################################
# Set up state variables to record weights, Gabor approximations, &
# losses that need to persist across loops.

def setup_state(opts, inputs):
    
    state = StateTuple(
        
        params=np.zeros((GABOR_NUM_PARAMS, opts.num_models),
                        dtype=np.float32),
        
        gabor=np.zeros(inputs.input_image.shape + (opts.num_models,),
                       dtype=np.float32),

        con_loss=np.zeros(opts.num_models, dtype=np.float32)

    )

    return state

######################################################################
# Perform a deep copy of a state

def copy_state(state):
    return StateTuple(*[x.copy() for x in state])

######################################################################
# Load weights from file.

def load_params(opts, inputs, models, state, sess):

    if opts.input is not None:
        iparams = np.genfromtxt(opts.input, dtype=np.float32, delimiter=',')
        nparams = len(iparams)
    else:
        iparams = np.empty((0, GABOR_NUM_PARAMS), dtype=np.float32)
        nparams = 0 

    print('loaded {} models from {}'.format(
        nparams, opts.input))

    nparams = min(nparams, opts.num_models)

    if nparams < len(iparams):
        print('warning: truncating input to {} models '
              'by randomly discarding {} models!'.format(
                  nparams, len(iparams)-nparams))
        idx = np.arange(len(iparams))
        np.random.shuffle(idx)
        iparams = iparams[idx[:nparams]]

    state.params[:,:nparams] = iparams.transpose()

    models.full.params.load(state.params[None,:,:], sess)

    fetches = dict(gabor=models.full.gabor,
                   approx=models.full.approx,
                   err_loss=models.full.err_loss,
                   loss=models.full.loss,
                   con_losses=models.full.con_losses)

    feed_dict = {inputs.target_tensor: inputs.input_image,
                 inputs.max_row: nparams}

    results = sess.run(fetches, feed_dict)

    state.gabor[:,:,:,:nparams] = results['gabor'][0, :, :, :, :nparams]
    state.con_loss[:nparams] = results['con_losses'][0, :nparams]

    cur_approx = results['approx'][0]

    prev_best_loss = results['err_loss'] + state.con_loss[:nparams].sum()
        
    if opts.preview_size:
        models.full.params.load(state.params[None,:,:])
    
    snapshot(None, cur_approx,
             opts, inputs, models, sess, -1, nparams, '')
    
    print('initial loss is {}'.format(prev_best_loss))
    print()

    model_start_idx = nparams

    return prev_best_loss, model_start_idx

######################################################################
# Rescale image to map given bounds to [0,255] uint8

def rescale(idata, imin, imax, cmap=None):

    assert imax > imin
    img = (idata - imin) / (imax - imin)
    img = np.clip(img, 0, 1)

    if cmap is not None:
        img = img.mean(axis=2)
        
    img = (img*255).astype(np.uint8)

    if cmap is not None:
        img = cmap[img]

    return img

######################################################################
# Save a snapshot of the current state to a PNG file

def snapshot(current_image, input_image, opts, iteration):
    """Save a snapshot of the current optimization state"""
    if not opts.snapshot_prefix:
        return
    
    try:
        # Create output directory if it doesn't exist
        outdir = os.path.dirname(opts.snapshot_prefix)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        
        # Generate output filename
        outfile = f'{opts.snapshot_prefix}-{iteration:06d}.png'
        print(f"Saving snapshot to: {outfile}")
        
        # Convert tensors to numpy arrays
        current_image_np = tf.cast(current_image * 255, tf.uint8).numpy()
        input_image_np = tf.cast(input_image * 255, tf.uint8).numpy()
        
        # Calculate error
        error = np.abs(current_image.numpy() - input_image.numpy())
        error_vis = tf.cast(error * 255, tf.uint8).numpy()
        
        # Create visualization
        h, w = input_image.shape[:2]
        canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        # Original image
        canvas[:, :w] = input_image_np
        # Current approximation
        canvas[:, w:w*2] = current_image_np
        # Error visualization
        canvas[:, w*2:] = error_vis
        
        # Add labels if requested
        if opts.label_snapshot:
            try:
                cv2.putText(canvas, 'Input', (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1,
                          (255, 255, 255), 2)
                cv2.putText(canvas, 'Current', (w + 10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1,
                          (255, 255, 255), 2)
                cv2.putText(canvas, 'Error', (w*2 + 10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1,
                          (255, 255, 255), 2)
            except ImportError:
                print("Warning: cv2 not available, skipping labels")
        
        # Save image
        try:
            imageio.imwrite(outfile, canvas)
            print(f"Successfully saved snapshot to {outfile}")
        except Exception as e:
            print(f"Failed to save snapshot: {e}")
            
    except Exception as e:
        print(f"Error in snapshot function: {e}")
        traceback.print_exc()

def add_snapshot_arguments(parser):
    """Add snapshot-related command line arguments"""
    parser.add_argument('--snapshot-prefix', type=str,
                       help='Prefix for snapshot filenames')
    parser.add_argument('--label-snapshot', action='store_true',
                       help='Add labels to snapshot images')

######################################################################

class GaborOptimizer:
    def __init__(self, model, input_image, learning_rate=0.01):
        self.model = model
        self.input_image = input_image
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.best_loss = float('inf')
        self.best_state = None

    @tf.function
    def optimization_step(self):
        """Single optimization step using gradient tape"""
        with tf.GradientTape() as tape:
            # Generate current approximation
            approx = self.model.generate_image()
            # Calculate loss (mean squared error)
            loss = tf.reduce_mean(tf.square(approx - self.input_image))
        
        # Get gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Apply constraints after optimization step
        self.model.apply_constraints()
        
        return loss, approx

class ProgressTracker:
    def __init__(self, total_iterations=None, time_limit=None):
        self.start_time = datetime.now()
        self.total_iterations = total_iterations
        self.time_limit = time_limit
        self.best_loss = float('inf')
        self.last_improvement = 0
        self.iteration = 0
        
        # Try to import tqdm for progress bars
        try:
            from tqdm import tqdm
            self.tqdm = tqdm
            self.use_progress_bar = True
        except ImportError:
            self.use_progress_bar = False
            print("Note: Install 'tqdm' for progress bar support")
    
    def start_iteration(self):
        """Start a new iteration"""
        self.iteration += 1
        if self.use_progress_bar:
            return self.tqdm(range(opts.steps_per_iteration),
                           desc=f"Iteration {self.iteration}")
        return range(opts.steps_per_iteration)
    
    def update_loss(self, current_loss):
        """Update best loss tracking"""
        improved = False
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.last_improvement = self.iteration
            improved = True
        return improved
    
    def should_continue(self):
        """Check if optimization should continue"""
        # Check time limit
        if self.time_limit is not None:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if elapsed > self.time_limit:
                print(f'\nExceeded time limit of {self.time_limit}s')
                return False
        
        # Check iteration limit
        if (self.total_iterations is not None and 
            self.iteration >= self.total_iterations):
            print(f'\nReached {self.total_iterations} iterations')
            return False
        
        return True
    
    def print_status(self, current_loss):
        """Print current optimization status"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        iterations_since_improvement = self.iteration - self.last_improvement
        
        print(f"\nIteration {self.iteration}:")
        print(f"  Current loss: {current_loss:.6f}")
        print(f"  Best loss: {self.best_loss:.6f}")
        print(f"  Time elapsed: {elapsed:.1f}s")
        print(f"  Iterations since improvement: {iterations_since_improvement}")
        
        if self.total_iterations:
            progress = self.iteration / self.total_iterations * 100
            print(f"  Progress: {progress:.1f}%")

def optimize_model(input_image, opts):
    """Main optimization function with progress tracking and helpers"""
    # Initialize model
    model = GaborModel('gabor', count=opts.num_gabors, 
                      image_shape=input_image.shape)
    
    # Load initial state if provided
    if opts.load_state:
        load_model_state(model, opts.load_state)
    
    optimizer = GaborOptimizer(model, input_image, 
                             learning_rate=opts.learning_rate)
    
    # Initialize progress tracker
    progress = ProgressTracker(opts.total_iterations, opts.time_limit)
    print(f"\nStarting optimization with {opts.num_gabors} Gabors")
    
    best_state = None
    last_save_time = datetime.now()
    
    while progress.should_continue():
        current_loss = None
        
        # Run optimization steps with progress bar
        for step in progress.start_iteration():
            # Get gradients and loss
            with tf.GradientTape() as tape:
                approx = model.generate_image()
                loss = tf.reduce_mean(tf.square(approx - input_image))
            
            # Get and clip gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            clipped_gradients = clip_gradients(gradients, opts.max_gradient_norm)
            
            # Apply gradients
            optimizer.optimizer.apply_gradients(
                zip(clipped_gradients, model.trainable_variables))
            
            # Apply constraints
            model.apply_constraints()
            
            current_loss = loss.numpy()
        
        # Update tracking and show status
        improved = progress.update_loss(current_loss)
        if improved:
            print("\n🌟 New best loss achieved!")
            best_state = model.get_variable_values()
            
            # Save intermediate best state
            if opts.save_best:
                save_model_state(model, 
                               f"{opts.save_best}.intermediate")
        
        # Print detailed status
        progress.print_status(current_loss)
        elapsed = (datetime.now() - progress.start_time).total_seconds()
        print(f"  Time elapsed: {format_time(elapsed)}")
        
        # Take snapshot if needed
        if opts.snapshot_prefix:
            try:
                snapshot(model.generate_image(), input_image, 
                        opts, progress.iteration)
            except Exception as e:
                print(f"⚠️  Warning: Failed to save snapshot: {e}")
        
        # Periodic state saving (every 10 minutes)
        if opts.save_best:
            time_since_save = (datetime.now() - last_save_time).total_seconds()
            if time_since_save > 600:  # 10 minutes
                save_model_state(model, 
                               f"{opts.save_best}.periodic")
                last_save_time = datetime.now()
        
        # Early stopping check
        if opts.early_stop:
            iterations_without_improvement = (
                progress.iteration - progress.last_improvement)
            if iterations_without_improvement > opts.patience:
                print(f"\n⚡ Early stopping: No improvement for "
                      f"{iterations_without_improvement} iterations")
                break
    
    # Restore best state
    if best_state is not None:
        model.set_variable_values(best_state)
    
    # Final save
    if opts.save_best:
        save_model_state(model, opts.save_best)
    
    print("\n✨ Optimization complete!")
    print(f"Final loss: {current_loss:.6f}")
    print(f"Best loss: {progress.best_loss:.6f}")
    print(f"Total time: {format_time(elapsed)}")
    
    return model, progress.best_loss

def add_optimization_arguments(parser):
    """Add optimization-related command line arguments"""
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--num-gabors', type=int, default=100,
                       help='Number of Gabor functions')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate for optimization')
    parser.add_argument('--early-stop', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=1000,
                       help='Number of iterations without improvement before early stopping')
    parser.add_argument('--total-iterations', type=int, default=100000,
                       help='Maximum number of iterations')
    parser.add_argument('--time-limit', type=float,
                       help='Time limit in seconds')
    parser.add_argument('--steps-per-iteration', type=int, default=1000,
                       help='Optimization steps per iteration')
    parser.add_argument('--max-gradient-norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')

def add_output_arguments(parser):
    """Add output-related command line arguments"""
    parser.add_argument('--output-dir', type=str,
                       help='Directory for output files')
    parser.add_argument('--save-best', type=str,
                       help='Save best model state to file')
    parser.add_argument('--load-state', type=str,
                       help='Load initial model state from file')
    parser.add_argument('--snapshot-prefix', type=str,
                       help='Prefix for snapshot filenames')
    parser.add_argument('--label-snapshot', action='store_true',
                       help='Add labels to snapshot images')

def setup_argument_parser():
    """Create and setup argument parser"""
    parser = argparse.ArgumentParser(description='TF2 Gabor Image Fitting')
    add_optimization_arguments(parser)
    add_output_arguments(parser)
    return parser

def main():
    # Setup argument parser
    parser = setup_argument_parser()
    opts = parser.parse_args()
    
    try:
        # Validate options
        validate_options(opts)
        
        # Create output directories
        create_output_directories(opts)
        
        # Load input image
        input_image = load_input_image(opts.input)
        
        # Run optimization
        model, final_loss = optimize_model(input_image, opts)
        
        # Save best model if requested
        if opts.save_best:
            save_model_state(model, opts.save_best)
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

def validate_options(opts):
    """Validate command line options"""
    if not os.path.exists(opts.input):
        raise ValueError(f"Input file not found: {opts.input}")
    
    if opts.num_gabors < 1:
        raise ValueError("Number of Gabors must be positive")
    
    if opts.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    
    if opts.time_limit is not None and opts.time_limit <= 0:
        raise ValueError("Time limit must be positive")
    
    if opts.total_iterations is not None and opts.total_iterations < 1:
        raise ValueError("Total iterations must be positive")

def create_output_directories(opts):
    """Create necessary output directories"""
    directories = []
    if opts.snapshot_prefix:
        directories.append(os.path.dirname(opts.snapshot_prefix))
    if opts.output_dir:
        directories.append(opts.output_dir)
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

def load_input_image(path):
    """Load and preprocess input image"""
    try:
        # Load image and normalize to [0, 1]
        image = imageio.imread(path).astype(np.float32) / 255.0
        # Ensure 3 channels (RGB)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:  # RGBA
            image = image[..., :3]
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to load image {path}: {e}")

def format_time(seconds):
    """Format time in seconds to human readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"
