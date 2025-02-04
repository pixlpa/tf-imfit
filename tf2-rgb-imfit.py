from __future__ import print_function
import re, os, sys, argparse
from datetime import datetime
from collections import namedtuple
import tensorflow as tf
import numpy as np
from PIL import Image

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
    [ 0, 0.05 ],
    [ 0, 0.05 ],
    [ 0, 0.05 ] ])

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
                        default=1000)

    parser.add_argument('-F', '--full-every', type=int, metavar='N',
                        default=32,
                        help='perform joint optimization '
                        'after every N outer loops')

    parser.add_argument('-f', '--full-iter', type=int, metavar='N',
                        help='maximum # of iterations for joint optimization',
                        default=1000)

    parser.add_argument('-r', '--local-learning-rate', type=float, metavar='R',
                        help='learning rate for local opt.',
                        default=0.001)

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
                        default=20)
     
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
  
    # Add optimization parameters
    parser.add_argument('--patience', type=int, default=10,
                       help='number of iterations to wait before adjusting learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-5,
                       help='minimum learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.5,
                       help='learning rate decay factor when progress stagnates')

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
        print('  {} {} {}'.format(opts.weights.name, weight_image.shape, weight_image.dtype))
        assert weight_image.size == input_image.size
    else:
        weight_image = 1.0

    # move to -1, 1 range for input image
    input_image = input_image * 2 - 1
    
    px, x, y = normalized_grid(input_image.shape[:2])

    GABOR_RANGE[GABOR_PARAM_L, 0] = 2.5*px
    GABOR_RANGE[GABOR_PARAM_T, 0] = px
    GABOR_RANGE[GABOR_PARAM_S, 0] = px
  
    target_tensor = tf.Variable(np.zeros(input_image.shape, dtype=np.float32),
                              trainable=False,
                              name='target')

    max_row = tf.Variable(0, trainable=False, dtype=tf.int32,
                        name='max_row')

    return InputsTuple(input_image, weight_image,
                       x, y, target_tensor, max_row)

######################################################################
# Encapsulate the tensorflow objects we need to run our fit.
# Note we will create several of these (see main function below).

class GaborModel(object):
 
    # The Gabor function tensor we define will be n x h x w x c x e
    # where n = num_parallel is the number of independent fits,
    # h x w is the image size,
    # c is the number of image channels (3 for RGB)
    # e = ensemble_size is the number of Gabor models per independent fit
    
    def __init__(self, 
                 num_parallel, ensemble_size,
                 x, y, weight, target,
                 learning_rate=0.0001,
                 params=None,
                 initializer=None,
                 max_row=None):
        
        # Store inputs
        self.x = x
        self.y = y
        self.weight = weight
        self.target = target
        self.max_row = ensemble_size if max_row is None else max_row
        
        # Initialize parameters
        if params is not None:
            # Convert params to Variable if it isn't already
            if not isinstance(params, tf.Variable):
                params = tf.Variable(params, trainable=True, dtype=tf.float32)
            self.params = params
        else:
            # Create random values directly using tf.random.uniform
            random_values = tf.random.uniform(
                shape=(num_parallel, GABOR_NUM_PARAMS, ensemble_size),
                dtype=tf.float32
            )

            # Scale and shift the random values to the desired ranges
            ranges = tf.cast(GABOR_RANGE[:, 1] - GABOR_RANGE[:, 0], tf.float32)
            mins = tf.cast(GABOR_RANGE[:, 0], tf.float32)

            # Reshape for broadcasting
            ranges = tf.reshape(ranges, [1, GABOR_NUM_PARAMS, 1])
            mins = tf.reshape(mins, [1, GABOR_NUM_PARAMS, 1])

            # Scale random values to proper ranges and create Variable
            self.params = tf.Variable(
                random_values * ranges + mins,
                trainable=True,
                dtype=tf.float32,
                name='params'
            )
        
        # Set up parameter ranges for clipping
        self.gmin = tf.constant(GABOR_RANGE[:,0], dtype=tf.float32)
        self.gmax = tf.constant(GABOR_RANGE[:,1], dtype=tf.float32)

        # Add learning rate tracking
        self.initial_lr = learning_rate
        self.min_lr = learning_rate * 0.01  # Minimum learning rate
        self.patience = 10  # Number of iterations before adjusting LR
        self.improvement_threshold = 1e-4  # Minimum meaningful improvement
        self.lr_decay = 0.5  # How much to reduce LR when plateauing
        
        # Add loss history tracking
        self.loss_history = []
        self.best_loss = float('inf')
        self.iterations_without_improvement = 0
        
        # Initialize optimizer with the initial learning rate
        self.opt = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0
        )
        
        # Initial forward pass
        self._forward_pass()

        # Check for NaN values
        if tf.reduce_any(tf.math.is_nan(self.params)):
            print("Warning: NaN values detected in params!")

        # Check input data shapes
        tf.print("Input image shape:", self.x.shape)
        tf.print("Input weight shape:", self.weight.shape)
        if self.target is not None:
            tf.print("Input target shape:", self.target.shape)

    def _forward_pass(self):
        with tf.name_scope('forward_pass'):
            # Reshape gmin and gmax for broadcasting
            gmin = tf.reshape(self.gmin, [1, GABOR_NUM_PARAMS, 1])
            gmax = tf.reshape(self.gmax, [1, GABOR_NUM_PARAMS, 1])
            
            # Clip parameters
            if tf.size(self.params) == 0:
                print("Warning: Parameters tensor is empty!")
            self.cparams = self.params
            
            # Extract parameters
            u = self.cparams[:,GABOR_PARAM_U,:]
            v = self.cparams[:,GABOR_PARAM_V,:]
            r = self.cparams[:,GABOR_PARAM_R,:]
            l = self.cparams[:,GABOR_PARAM_L,:]
            t = self.cparams[:,GABOR_PARAM_T,:]
            s = self.cparams[:,GABOR_PARAM_S,:]
            
            # Extract RGB parameters
            h = self.cparams[:,GABOR_PARAM_H0:GABOR_PARAM_H0+3,:]  # [batch, 3, models]
            p = self.cparams[:,GABOR_PARAM_P0:GABOR_PARAM_P0+3,:]  # [batch, 3, models]
            
            # Add necessary dimensions for broadcasting
            u = u[:,None,None,None,:]  # [batch, 1, 1, 1, models]
            v = v[:,None,None,None,:]
            r = r[:,None,None,None,:]
            l = l[:,None,None,None,:]
            t = t[:,None,None,None,:]
            s = s[:,None,None,None,:]
            h = h[:,None,None,:,:]  # [batch, 1, 1, 3, models]
            p = p[:,None,None,:,:]
            
            # Compute Gabor function
            cr = tf.cos(r)
            sr = tf.sin(r)
            f = tf.cast(2*np.pi, tf.float32) / l
            s2 = s*s
            t2 = t*t
            
            xp = self.x - u
            yp = self.y - v
            
            b1 = cr*xp + sr*yp
            b2 = -sr*xp + cr*yp
            
            b12 = b1*b1
            b22 = b2*b2
            
            exp_term = -b12/(2*s2) - b22/(2*t2)
            w = tf.exp(exp_term)  # Gaussian envelope [0,1]
            
            k = f*b1 + p
            ck = tf.cos(k)  # Oscillation [-1,1]
            
            # Combine components with proper scaling for RGB
            self.gabor = tf.identity(h * w * ck, name='gabor')
            self.approx = tf.reduce_sum(self.gabor, axis=4, name='approx')
            
            if self.target is not None:
                self._compute_losses()

    def _compute_losses(self):
        """Compute losses with numerical safeguards"""
        # Compute error loss
        self.err = tf.multiply((self.target - self.approx), self.weight)
        err_sqr = 0.5 * self.err**2
        
        # Per-fit error losses (average across h/w/c)
        self.err_loss_per_fit = tf.reduce_mean(err_sqr, axis=(1,2,3))
        # Overall error loss
        self.err_loss = tf.reduce_mean(self.err_loss_per_fit)
        
        # Compute constraints
        l = self.cparams[:,GABOR_PARAM_L,:]
        s = self.cparams[:,GABOR_PARAM_S,:]
        t = self.cparams[:,GABOR_PARAM_T,:]
        
        constraints = [
            s - l/32,
            l/2 - s,
            t - s,
            8*s - t
        ]
        
        # Stack constraints (n x e x k)
        self.constraints = tf.stack(constraints, axis=2)
        
        # Compute squared constraints (n x e x k)
        con_sqr = tf.minimum(self.constraints, 0)**2
        
        # Per-model constraint losses (n x e)
        self.con_losses = tf.reduce_sum(con_sqr, axis=2)
        
        # Per-fit constraint losses (n)
        self.con_loss_per_fit = tf.reduce_sum(self.con_losses, axis=1)
        
        # Overall constraint loss
        self.con_loss = tf.reduce_mean(self.con_loss_per_fit)
        
        # Total loss per fit (n)
        self.loss_per_fit = self.err_loss_per_fit + self.con_loss_per_fit
        
        # Overall total loss
        self.loss = self.err_loss + self.con_loss

    @tf.function
    def train_step(self):
        """Performs one training step with adaptive learning rate"""
        with tf.GradientTape() as tape:
            self._forward_pass()
            loss = self.loss
            
        # Get gradients
        gradients = tape.gradient(loss, [self.params])
        gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
        
        # Apply gradients if they exist and are not NaN
        if gradients[0] is not None:
            self.opt.apply_gradients(zip(gradients, [self.params]))
            
        # Add random perturbation if we're stuck
        if self.iterations_without_improvement > self.patience * 2:
            noise = tf.random.normal(self.params.shape, stddev=0.01)
            self.params.assign_add(noise)
            self.iterations_without_improvement = 0
            
        return {
            'loss': loss,
            'gabor': self.gabor,
            'approx': self.approx,
            'params': self.params,
            'err_loss_per_fit': self.err_loss_per_fit,
            'con_loss_per_fit': self.con_loss_per_fit
        }

    def adjust_learning_rate(self, current_loss):
        """Adjust learning rate based on loss improvement"""
        self.loss_history.append(current_loss)
        
        # Check if this is the best loss we've seen
        if current_loss < self.best_loss * (1 - self.improvement_threshold):
            self.best_loss = current_loss
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
        
        # If we haven't improved for patience iterations, reduce learning rate
        if self.iterations_without_improvement >= self.patience:
            current_lr = self.opt.learning_rate.numpy()
            if current_lr > self.min_lr:
                new_lr = max(current_lr * self.lr_decay, self.min_lr)
                self.opt.learning_rate.assign(new_lr)
                print(f"Reducing learning rate to {new_lr:.6f}")
                self.iterations_without_improvement = 0
                
        return self.iterations_without_improvement

    def reset_optimization(self):
        """Reset optimization state"""
        self.opt.learning_rate.assign(self.initial_lr)
        self.loss_history = []
        self.best_loss = float('inf')
        self.iterations_without_improvement = 0

    def get_current_state(self):
        """Get current model state without using @tf.function"""
        self._forward_pass()
        return {
            'loss': self.loss,
            'gabor': self.gabor,
            'approx': self.approx,
            'params': self.params,
            'err_loss': self.err_loss,
            'err_loss_per_fit': self.err_loss_per_fit,
            'con_losses': self.con_losses,
            'con_loss_per_fit': self.con_loss_per_fit
        }

######################################################################
# Set up tensorflow models themselves. We need a separate model for
# each combination of inputs/dimensions to optimize.

def setup_models(opts, inputs, state):
    # Validate dimensions
    if opts.num_models <= 0:
        raise ValueError("num_models must be positive")
    if opts.num_local <= 0:
        raise ValueError("num_local must be positive")
        
    weight_tensor = tf.constant(inputs.weight_image)

    x_tensor = tf.constant(inputs.x.reshape(1,1,-1,1,1))
    y_tensor = tf.constant(inputs.y.reshape(1,-1,1,1,1))

    # Check target tensor before using it
    if inputs.target_tensor is None:
        print("Error: inputs.target_tensor is None!")
    else:
        tf.print("Target tensor shape:", inputs.target_tensor.shape)

    with tf.name_scope('full'):
        full = GaborModel(1, opts.num_models,
                          x_tensor, y_tensor,
                          weight_tensor, inputs.target_tensor,
                          learning_rate=opts.full_learning_rate,
                          max_row=inputs.max_row,
                          params=tf.Variable(state.params[None,:], trainable=True))
    
    with tf.name_scope('local'):
        
        local = GaborModel(opts.num_local, 1,
                           x_tensor, y_tensor,
                           weight_tensor, inputs.target_tensor,
                           learning_rate=opts.local_learning_rate)
        

    if opts.preview_size:

        preview_shape = scale_shape(map(int, inputs.target_tensor.shape[:2]),
                                    opts.preview_size)

        _, x_preview, y_preview = normalized_grid(preview_shape[:2])
        
        with tf.name_scope('preview'):
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
    # Initialize parameters randomly within their valid ranges
    params = np.zeros((GABOR_NUM_PARAMS, opts.num_models), dtype=np.float32)
    for i in range(GABOR_NUM_PARAMS):
        if GABOR_PARAM_H0 <= i < GABOR_PARAM_H0+3:
            # For h parameters, create a broader dynamic range via a normal distribution.
            center = (GABOR_RANGE[i, 0] + GABOR_RANGE[i, 1]) / 2.0
            range_width = GABOR_RANGE[i, 1] - GABOR_RANGE[i, 0]
            # Adjust the scale factor (here, range_width/2) to be larger if needed.
            params[i] = np.clip(
                np.random.normal(loc=center, scale=range_width/2.0, size=opts.num_models),
                GABOR_RANGE[i, 0],
                GABOR_RANGE[i, 1]
            )
        else:
            params[i] = np.random.uniform(
                GABOR_RANGE[i, 0],
                GABOR_RANGE[i, 1],
                opts.num_models
            ) 
    state = StateTuple(
        params=params,
        gabor=np.zeros(inputs.input_image.shape + (opts.num_models,),
                       dtype=np.float32),
        con_loss=np.zeros(opts.num_models, dtype=np.float32))
    return state

######################################################################
# Perform a deep copy of a state

def copy_state(state):
    return StateTuple(*[x.copy() for x in state])

######################################################################
# Load weights from file.

def load_params(opts, inputs, models, state):
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

    # Update the full model's parameters using tf.Variable.assign
    new_params = tf.constant(state.params[None,:], dtype=tf.float32)
    models.full.params.assign(new_params)
 
    # Set the target and max_row
    inputs.target_tensor.assign(inputs.input_image)
    inputs.max_row.assign(nparams)

    # Force a forward pass to compute initial error
    _ = models.full._forward_pass()
    
    # Get the results directly from the model
    gabor = models.full.gabor.numpy()[0]  # Get gabor values
    approx = models.full.approx.numpy()[0]
    err_loss = float(models.full.err_loss)
    con_losses = models.full.con_losses.numpy()[0]

    # For initial case with no models, err_loss should be the MSE between target and zeros
    if nparams == 0:
        err_loss = float(tf.reduce_mean(inputs.input_image ** 2))
        print(f"Initial error (no models): {err_loss}")

    # Update state with results
    state.gabor[:,:,:,:nparams] = gabor[:,:,:,:nparams]
    state.con_loss[:nparams] = con_losses[:nparams]

    prev_best_loss = err_loss + state.con_loss[:nparams].sum()
    
    if opts.preview_size:
        models.preview.params.assign(new_params)
    
    # Pass gabor values to snapshot instead of None
    snapshot(gabor, approx,
             opts, inputs, models,
             -1, '')
    
    print('initial loss is {}'.format(prev_best_loss))
    print()

    model_start_idx = nparams

    # After creating/assigning models.full.params, test sensitivity:
    with tf.GradientTape() as tape:
        tape.watch(models.full.params)
        # Instead of a full forward pass, just compute a simple function of self.params:
        dummy_output = tf.reduce_sum(models.full.params * 2.0)
    grad_dummy = tape.gradient(dummy_output, models.full.params)
    print("Dummy grad norm (should not be 0):", tf.linalg.global_norm([grad_dummy]).numpy())

    return prev_best_loss, model_start_idx

######################################################################
# Rescale image to map given bounds to [0,255] uint8

def rescale(img, src_min, src_max, colormap=None):
    """
    Rescale image from [src_min,src_max] to [0,255] range.
    If colormap provided, use it for the output instead of grayscale.
    """
    # Ensure float32 for calculations
    img = img.astype(np.float32)
    
    # Scale to [0,1] range
    if src_max != src_min:
        img = (img - src_min) / (src_max - src_min)
    else:
        img = np.zeros_like(img)
    
    # Clip to ensure we're in [0,1]
    img = np.clip(img, 0, 1)
    
    # Scale to [0,255]
    img = (img * 255).astype(np.uint8)
    
    if colormap is not None:
        # Convert to RGB using colormap
        if len(img.shape) == 3:
            # Convert each channel separately
            result = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
            for i in range(3):
                result[..., i] = colormap[img[..., i]][..., i]
            return result
        else:
            # Single channel image
            return colormap[img]
    
    return img

######################################################################
# Save a snapshot of the current state to a PNG file

def snapshot(cur_gabor, cur_approx,
             opts, inputs, models,
             loop_count, 
             full_iteration):
    """
    Save a snapshot of the current state to a PNG file.
    
    The output montage includes:
      - The target image,
      - The full approximation from all active filters,
      - The gabor image,
      - An absolute error visualization (optionally enhanced).
    """
    if not opts.label_snapshot:
        outfile = '{}.png'.format(opts.snapshot_prefix)
    elif isinstance(full_iteration, int):
        outfile = '{}{:04d}_{:06d}.png'.format(
            opts.snapshot_prefix, loop_count + 1, full_iteration + 1)
    else:
        outfile = '{}{:04d}{}.png'.format(
            opts.snapshot_prefix, loop_count + 1, full_iteration)

    if cur_gabor is None or cur_gabor.size == 0:
        print("Creating zero gabor array")
        cur_gabor = np.zeros_like(cur_approx)
    
    # Compute the residual image.
    # This is the signed difference between the target (inputs.input_image)
    # and the full approximation (cur_approx) from all active models.
    residual = inputs.input_image - cur_approx

    # Compute an absolute error image for visualization purposes.
    # Here, we use a non-linear transform to boost low differences.
    cur_abserr = np.abs(residual)
    cur_abserr = cur_abserr * inputs.weight_image
    cur_abserr = np.power(cur_abserr, 0.5)

    global COLORMAP
    if COLORMAP is None:
        COLORMAP = get_colormap()
    
    # Rescale images for display:
    # We assume that both the target and the approximation are in [-1, 1].
    input_img    = rescale(inputs.input_image, -1, 1)
    approx_img   = rescale(cur_approx, -1, 1)
    error_img    = rescale(cur_abserr, 0, cur_abserr.max(), COLORMAP)

    # Create a montage of the images.
    # Montage order: Target | Approximation | Residual | Absolute Error
    out_img = np.hstack((input_img, approx_img, error_img))
    
    out_img = Image.fromarray(out_img.astype(np.uint8), 'RGB')
    out_img.save(outfile)

######################################################################
# Perform an optimization on the full joint model (expensive/slow).

def full_optimize(opts, inputs, models, state, start_idx, loop_count, prev_best_loss, optimizer):
    """Optimize all models' parameters using full optimization."""
    
    # Get initial loss
    _ = models.full._forward_pass()
    loss = models.full.err_loss + tf.reduce_sum(models.full.con_losses)
    print(f"  loss before full optimization is {float(loss):.9f}")
    
    with tf.GradientTape() as tape:
        # Ensure that the tape watches the model parameters
        tape.watch(models.full.params)
        _ = models.full._forward_pass()
        err_loss = models.full.err_loss
        con_loss = tf.reduce_sum(models.full.con_losses)
        total_loss = err_loss + con_loss
        
    # Compute gradients
    grads = tape.gradient(total_loss, [models.full.params])
    grad_norm = tf.linalg.global_norm(grads)
    print("  Grad norm:", grad_norm.numpy())
    
    if grads[0] is not None and grad_norm.numpy() > 1e-8:
        optimizer.apply_gradients(zip(grads, [models.full.params]))
        
        # Update preview and snapshot if needed
        if opts.preview_size:
            models.preview.params.assign(models.full.params)
            _ = models.preview._forward_pass()
        
        state_dict = models.full.get_current_state()
        gabor = state_dict['gabor'].numpy()[0]  # Remove batch dimension
        approx = state_dict['approx'].numpy()[0]  # Remove batch dimension
        
        print(f"  loss after full optimization is {float(total_loss):.9f}")
    else:
        print("  Gradients are zero, not applying update.")
        
    return float(total_loss)

######################################################################
# Apply a small perturbation to the input parameters

def randomize(params, rstdev, ncopy=None):

    gmin = GABOR_RANGE[:,0]
    gmax = GABOR_RANGE[:,1]
    grng = gmax - gmin

    pshape = params.shape

    if ncopy is not None:
        pshape = (ncopy,) + pshape
    
    bump = np.random.normal(scale=rstdev, size=pshape)

    
    return params + bump*grng

######################################################################
# Optimize a bunch of randomly-initialized small ensembles in
# parallel.

def local_optimize(opts, inputs, models, state, current_model, loop_count):
    """Optimize a single model's parameters using local optimization."""
    for i in range(opts.local_iter):
        # Get initial loss
        _ = models.local._forward_pass()
        loss = models.local.err_loss + models.local.con_losses[0]
        # print(f"  loss before local fit is {float(loss):.9f}")
        
        # Single optimization step with Adam
        with tf.GradientTape() as tape:
            tape.watch(models.local.params)
            _ = models.local._forward_pass()
            err_loss = models.local.err_loss
            con_loss = models.local.con_losses[0]
            total_loss = err_loss + con_loss
            
        # Compute and apply gradients
        grads = tape.gradient(total_loss, [models.local.params])
        if grads[0] is not None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=opts.local_learning_rate)
            optimizer.apply_gradients(zip(grads, [models.local.params]))
            
            # Update preview and snapshot
            if opts.preview_size:
                models.preview.params.assign(models.full.params)
                _ = models.preview._forward_pass()
            

    # Get the results directly from the model
    gabor = models.full.gabor.numpy()[0]  # Get gabor values
    approx = models.full.approx.numpy()[0]

    # Convert tensors to numpy arrays and handle dimensions
    snapshot(gabor, approx,
            opts, inputs, models,
            loop_count, '')

    print(f"  loss after local fit is {float(total_loss):.9f}")
    
    return float(total_loss)

######################################################################

def main():
    # Set up command-line options, image inputs, initial state, and models.
    opts = get_options()
    inputs = setup_inputs(opts)
    state = setup_state(opts, inputs)
    models = setup_models(opts, inputs, state)

    # Set the target to the original input image.
    inputs.target_tensor.assign(inputs.input_image)

    # If initial parameter files were provided, load them.
    prev_best_loss, initial_filter_count = load_params(opts, inputs, models, state)

    # Starting filter count is the number already in the state.
    current_model = initial_filter_count
    # Create the optimizer for the full optimization phase.
    optimizer = tf.keras.optimizers.Adam(learning_rate=opts.full_learning_rate)

    loop_count = 0
    start_time = datetime.now()

    try:
        while True:
            # Exit if time or iteration limits are reached.
            if opts.time_limit is not None:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > opts.time_limit:
                    print('Exceeded time limit, quitting!')
                    break
            if opts.total_iterations is not None and loop_count >= opts.total_iterations:
                print(f"Reached {opts.total_iterations} iterations, quitting!")
                break

            print(f"Iteration #{loop_count+1}:")

            # -----------------------------
            # 1. Compute the residual image.
            # -----------------------------
            if current_model > 0:
                # Sum the output of all filters added so far.
                current_approx = state.gabor[:,:,:,:current_model].sum(axis=3)
                residual = inputs.input_image - current_approx
            else:
                # If no model has been added yet, the residual is the full image.
                residual = inputs.input_image

            # For local optimization we want to fit the candidate filter
            # to the residual (i.e. the part not explained by earlier filters).
            inputs.target_tensor.assign(residual)

            # -----------------------------
            # 2. Local (per-filter) optimization.
            # -----------------------------
            # Optimize a single candidate filter against the residual image.
            print(f"Performing local optimization for filter {current_model+1} / {opts.num_models}")
            local_loss = local_optimize(opts, inputs, models, state, current_model, loop_count)
            print(f"Local loss (filter {current_model+1}): {local_loss:.9f}")

            # Assume that local_optimize (directly or via snapshot())
            # updates state.gabor and state.params for the candidate filter.
            current_model += 1

            # -----------------------------
            # 3. Occasional full (joint) optimization.
            # -----------------------------
            if (loop_count+1) % opts.full_every == 0:
                print("Performing full joint optimization over all filters...")
                prev_best_loss = full_optimize(opts, inputs, models, state,
                                               current_model, loop_count, prev_best_loss, optimizer)
                if opts.output is not None:
                    np.savetxt(opts.output, state.params.transpose(),
                               fmt='%f', delimiter=',')

            loop_count += 1

    except KeyboardInterrupt:
        print("Interrupted. Saving final state and exiting...")
        if opts.output is not None:
                    np.savetxt(opts.output, state.params.transpose(),
                               fmt='%f', delimiter=',')
    if opts.output is not None:
        np.savetxt(opts.output, state.params.transpose(), fmt='%f', delimiter=',')
######################################################################

# from https://github.com/BIDS/colormap/blob/master/colormaps.py
# licensed CC0
_magma_data = [[0.001462, 0.000466, 0.013866],
               [0.002258, 0.001295, 0.018331],
               [0.003279, 0.002305, 0.023708],
               [0.004512, 0.003490, 0.029965],
               [0.005950, 0.004843, 0.037130],
               [0.007588, 0.006356, 0.044973],
               [0.009426, 0.008022, 0.052844],
               [0.011465, 0.009828, 0.060750],
               [0.013708, 0.011771, 0.068667],
               [0.016156, 0.013840, 0.076603],
               [0.018815, 0.016026, 0.084584],
               [0.021692, 0.018320, 0.092610],
               [0.024792, 0.020715, 0.100676],
               [0.028123, 0.023201, 0.108787],
               [0.031696, 0.025765, 0.116965],
               [0.035520, 0.028397, 0.125209],
               [0.039608, 0.031090, 0.133515],
               [0.043830, 0.033830, 0.141886],
               [0.048062, 0.036607, 0.150327],
               [0.052320, 0.039407, 0.158841],
               [0.056615, 0.042160, 0.167446],
               [0.060949, 0.044794, 0.176129],
               [0.065330, 0.047318, 0.184892],
               [0.069764, 0.049726, 0.193735],
               [0.074257, 0.052017, 0.202660],
               [0.078815, 0.054184, 0.211667],
               [0.083446, 0.056225, 0.220755],
               [0.088155, 0.058133, 0.229922],
               [0.092949, 0.059904, 0.239164],
               [0.097833, 0.061531, 0.248477],
               [0.102815, 0.063010, 0.257854],
               [0.107899, 0.064335, 0.267289],
               [0.113094, 0.065492, 0.276784],
               [0.118405, 0.066479, 0.286321],
               [0.123833, 0.067295, 0.295879],
               [0.129380, 0.067935, 0.305443],
               [0.135053, 0.068391, 0.315000],
               [0.140858, 0.068654, 0.324538],
               [0.146785, 0.068738, 0.334011],
               [0.152839, 0.068637, 0.343404],
               [0.159018, 0.068354, 0.352688],
               [0.165308, 0.067911, 0.361816],
               [0.171713, 0.067305, 0.370771],
               [0.178212, 0.066576, 0.379497],
               [0.184801, 0.065732, 0.387973],
               [0.191460, 0.064818, 0.396152],
               [0.198177, 0.063862, 0.404009],
               [0.204935, 0.062907, 0.411514],
               [0.211718, 0.061992, 0.418647],
               [0.218512, 0.061158, 0.425392],
               [0.225302, 0.060445, 0.431742],
               [0.232077, 0.059889, 0.437695],
               [0.238826, 0.059517, 0.443256],
               [0.245543, 0.059352, 0.448436],
               [0.252220, 0.059415, 0.453248],
               [0.258857, 0.059706, 0.457710],
               [0.265447, 0.060237, 0.461840],
               [0.271994, 0.060994, 0.465660],
               [0.278493, 0.061978, 0.469190],
               [0.284951, 0.063168, 0.472451],
               [0.291366, 0.064553, 0.475462],
               [0.297740, 0.066117, 0.478243],
               [0.304081, 0.067835, 0.480812],
               [0.310382, 0.069702, 0.483186],
               [0.316654, 0.071690, 0.485380],
               [0.322899, 0.073782, 0.487408],
               [0.329114, 0.075972, 0.489287],
               [0.335308, 0.078236, 0.491024],
               [0.341482, 0.080564, 0.492631],
               [0.347636, 0.082946, 0.494121],
               [0.353773, 0.085373, 0.495501],
               [0.359898, 0.087831, 0.496778],
               [0.366012, 0.090314, 0.497960],
               [0.372116, 0.092816, 0.499053],
               [0.378211, 0.095332, 0.500067],
               [0.384299, 0.097855, 0.501002],
               [0.390384, 0.100379, 0.501864],
               [0.396467, 0.102902, 0.502658],
               [0.402548, 0.105420, 0.503386],
               [0.408629, 0.107930, 0.504052],
               [0.414709, 0.110431, 0.504662],
               [0.420791, 0.112920, 0.505215],
               [0.426877, 0.115395, 0.505714],
               [0.432967, 0.117855, 0.506160],
               [0.439062, 0.120298, 0.506555],
               [0.445163, 0.122724, 0.506901],
               [0.451271, 0.125132, 0.507198],
               [0.457386, 0.127522, 0.507448],
               [0.463508, 0.129893, 0.507652],
               [0.469640, 0.132245, 0.507809],
               [0.475780, 0.134577, 0.507921],
               [0.481929, 0.136891, 0.507989],
               [0.488088, 0.139186, 0.508011],
               [0.494258, 0.141462, 0.507988],
               [0.500438, 0.143719, 0.507920],
               [0.506629, 0.145958, 0.507806],
               [0.512831, 0.148179, 0.507648],
               [0.519045, 0.150383, 0.507443],
               [0.525270, 0.152569, 0.507192],
               [0.531507, 0.154739, 0.506895],
               [0.537755, 0.156894, 0.506551],
               [0.544015, 0.159033, 0.506159],
               [0.550287, 0.161158, 0.505719],
               [0.556571, 0.163269, 0.505230],
               [0.562866, 0.165368, 0.504692],
               [0.569172, 0.167454, 0.504105],
               [0.575490, 0.169530, 0.503466],
               [0.581819, 0.171596, 0.502777],
               [0.588158, 0.173652, 0.502035],
               [0.594508, 0.175701, 0.501241],
               [0.600868, 0.177743, 0.500394],
               [0.607238, 0.179779, 0.499492],
               [0.613617, 0.181811, 0.498536],
               [0.620005, 0.183840, 0.497524],
               [0.626401, 0.185867, 0.496456],
               [0.632805, 0.187893, 0.495332],
               [0.639216, 0.189921, 0.494150],
               [0.645633, 0.191952, 0.492910],
               [0.652056, 0.193986, 0.491611],
               [0.658483, 0.196027, 0.490253],
               [0.664915, 0.198075, 0.488836],
               [0.671349, 0.200133, 0.487358],
               [0.677786, 0.202203, 0.485819],
               [0.684224, 0.204286, 0.484219],
               [0.690661, 0.206384, 0.482558],
               [0.697098, 0.208501, 0.480835],
               [0.703532, 0.210638, 0.479049],
               [0.709962, 0.212797, 0.477201],
               [0.716387, 0.214982, 0.475290],
               [0.722805, 0.217194, 0.473316],
               [0.729216, 0.219437, 0.471279],
               [0.735616, 0.221713, 0.469180],
               [0.742004, 0.224025, 0.467018],
               [0.748378, 0.226377, 0.464794],
               [0.754737, 0.228772, 0.462509],
               [0.761077, 0.231214, 0.460162],
               [0.767398, 0.233705, 0.457755],
               [0.773695, 0.236249, 0.455289],
               [0.779968, 0.238851, 0.452765],
               [0.786212, 0.241514, 0.450184],
               [0.792427, 0.244242, 0.447543],
               [0.798608, 0.247040, 0.444848],
               [0.804752, 0.249911, 0.442102],
               [0.810855, 0.252861, 0.439305],
               [0.816914, 0.255895, 0.436461],
               [0.822926, 0.259016, 0.433573],
               [0.828886, 0.262229, 0.430644],
               [0.834791, 0.265540, 0.427671],
               [0.840636, 0.268953, 0.424666],
               [0.846416, 0.272473, 0.421631],
               [0.852126, 0.276106, 0.418573],
               [0.857763, 0.279857, 0.415496],
               [0.863320, 0.283729, 0.412403],
               [0.868793, 0.287728, 0.409303],
               [0.874176, 0.291859, 0.406205],
               [0.879464, 0.296125, 0.403118],
               [0.884651, 0.300530, 0.400047],
               [0.889731, 0.305079, 0.397002],
               [0.894700, 0.309773, 0.393995],
               [0.899552, 0.314616, 0.391037],
               [0.904281, 0.319610, 0.388137],
               [0.908884, 0.324755, 0.385308],
               [0.913354, 0.330052, 0.382563],
               [0.917689, 0.335500, 0.379915],
               [0.921884, 0.341098, 0.377376],
               [0.925937, 0.346844, 0.374959],
               [0.929845, 0.352734, 0.372677],
               [0.933606, 0.358764, 0.370541],
               [0.937221, 0.364929, 0.368567],
               [0.940687, 0.371224, 0.366762],
               [0.944006, 0.377643, 0.365136],
               [0.947180, 0.384178, 0.363701],
               [0.950210, 0.390820, 0.362468],
               [0.953099, 0.397563, 0.361438],
               [0.955849, 0.404400, 0.360619],
               [0.958464, 0.411324, 0.360014],
               [0.960949, 0.418323, 0.359630],
               [0.963310, 0.425390, 0.359469],
               [0.965549, 0.432519, 0.359529],
               [0.967671, 0.439703, 0.359810],
               [0.969680, 0.446936, 0.360311],
               [0.971582, 0.454210, 0.361030],
               [0.973381, 0.461520, 0.361965],
               [0.975082, 0.468861, 0.363111],
               [0.976690, 0.476226, 0.364466],
               [0.978210, 0.483612, 0.366025],
               [0.979645, 0.491014, 0.367783],
               [0.981000, 0.498428, 0.369734],
               [0.982279, 0.505851, 0.371874],
               [0.983485, 0.513280, 0.374198],
               [0.984622, 0.520713, 0.376698],
               [0.985693, 0.528148, 0.379371],
               [0.986700, 0.535582, 0.382210],
               [0.987646, 0.543015, 0.385210],
               [0.988533, 0.550446, 0.388365],
               [0.989363, 0.557873, 0.391671],
               [0.990138, 0.565296, 0.395122],
               [0.990871, 0.572706, 0.398714],
               [0.991558, 0.580107, 0.402441],
               [0.992196, 0.587502, 0.406299],
               [0.992785, 0.594891, 0.410283],
               [0.993326, 0.602275, 0.414390],
               [0.993834, 0.609644, 0.418613],
               [0.994309, 0.616999, 0.422950],
               [0.994738, 0.624350, 0.427397],
               [0.995122, 0.631696, 0.431951],
               [0.995480, 0.639027, 0.436607],
               [0.995810, 0.646344, 0.441361],
               [0.996096, 0.653659, 0.446213],
               [0.996341, 0.660969, 0.451160],
               [0.996580, 0.668256, 0.456192],
               [0.996775, 0.675541, 0.461314],
               [0.996925, 0.682828, 0.466526],
               [0.997077, 0.690088, 0.471811],
               [0.997186, 0.697349, 0.477182],
               [0.997254, 0.704611, 0.482635],
               [0.997325, 0.711848, 0.488154],
               [0.997351, 0.719089, 0.493755],
               [0.997351, 0.726324, 0.499428],
               [0.997341, 0.733545, 0.505167],
               [0.997285, 0.740772, 0.510983],
               [0.997228, 0.747981, 0.516859],
               [0.997138, 0.755190, 0.522806],
               [0.997019, 0.762398, 0.528821],
               [0.996898, 0.769591, 0.534892],
               [0.996727, 0.776795, 0.541039],
               [0.996571, 0.783977, 0.547233],
               [0.996369, 0.791167, 0.553499],
               [0.996162, 0.798348, 0.559820],
               [0.995932, 0.805527, 0.566202],
               [0.995680, 0.812706, 0.572645],
               [0.995424, 0.819875, 0.579140],
               [0.995131, 0.827052, 0.585701],
               [0.994851, 0.834213, 0.592307],
               [0.994524, 0.841387, 0.598983],
               [0.994222, 0.848540, 0.605696],
               [0.993866, 0.855711, 0.612482],
               [0.993545, 0.862859, 0.619299],
               [0.993170, 0.870024, 0.626189],
               [0.992831, 0.877168, 0.633109],
               [0.992440, 0.884330, 0.640099],
               [0.992089, 0.891470, 0.647116],
               [0.991688, 0.898627, 0.654202],
               [0.991332, 0.905763, 0.661309],
               [0.990930, 0.912915, 0.668481],
               [0.990570, 0.920049, 0.675675],
               [0.990175, 0.927196, 0.682926],
               [0.989815, 0.934329, 0.690198],
               [0.989434, 0.941470, 0.697519],
               [0.989077, 0.948604, 0.704863],
               [0.988717, 0.955742, 0.712242],
               [0.988367, 0.962878, 0.719649],
               [0.988033, 0.970012, 0.727077],
               [0.987691, 0.977154, 0.734536],
               [0.987387, 0.984288, 0.742002],
               [0.987053, 0.991438, 0.749504]]    

def get_colormap():
    return (np.array(_magma_data)*255).astype(np.uint8)

######################################################################

if __name__ == '__main__':
    main()
