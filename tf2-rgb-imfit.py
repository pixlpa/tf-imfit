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
            
        # Pre-compute coordinate grid once during initialization
        x_coords = tf.cast(tf.range(w), tf.float32)
        y_coords = tf.cast(tf.range(h), tf.float32)
        self.x_grid, self.y_grid = tf.meshgrid(x_coords, y_coords)
        
        print(f"Initialized GaborModel with {count} Gabors for image shape {image_shape}")

    @property
    def trainable_variables(self):
        """Return list of all trainable variables"""
        return list(self.variables.values())

    @tf.function
    def generate_gabor(self, x, y):
        """Generate Gabor function values for given coordinates"""
        # Use broadcasting for better performance
        x = tf.expand_dims(x, -1)  # [..., 1]
        y = tf.expand_dims(y, -1)  # [..., 1]
        
        # Center coordinates on Gabor positions (use broadcasting)
        x_c = x - self.pos_x  # [..., count]
        y_c = y - self.pos_y  # [..., count]
        
        # Pre-compute trig functions once
        cos_theta = tf.cos(self.theta)
        sin_theta = tf.sin(self.theta)
        
        # Rotate coordinates (vectorized)
        x_r = x_c * cos_theta + y_c * sin_theta
        y_r = -x_c * sin_theta + y_c * cos_theta
        
        # Compute Gabor function (vectorized)
        gaussian = tf.exp(-(x_r**2 + y_r**2) / (2 * self.sigma**2))
        sinusoid = tf.cos(2 * np.pi * self.frequency * x_r + self.phase)
        
        return gaussian * sinusoid * self.amplitude * self.scale

    @tf.function
    def generate_image(self):
        """Generate full image from all Gabor functions (optimized)"""
        # Use pre-computed coordinate grid
        gabor_values = self.generate_gabor(self.x_grid, self.y_grid)  # [h, w, count]
        
        # Multiply by colors and sum (optimized broadcasting)
        colored_gabors = tf.expand_dims(gabor_values, -1) * self.colors  # [h, w, count, 3]
        image = tf.reduce_sum(colored_gabors, axis=2)  # [h, w, 3]
        
        return tf.clip_by_value(image, 0.0, 1.0)

    def apply_constraints(self):
        """Apply constraints to keep variables in valid ranges"""
        h, w = self.image_shape[:2]
        constraints = {
            'pos_x': (0, w),
            'pos_y': (0, h),
            'sigma': (1, 20),
            'theta': (0, np.pi),
            'frequency': (0.02, 0.5),
            'phase': (0, 2*np.pi),
            'colors': (0, 1),
            'amplitude': (0, 1),
            'scale': (0, None)
        }
        
        for name, (min_val, max_val) in constraints.items():
            var = self.variables[name]
            if min_val is not None:
                var.assign(tf.maximum(var, min_val))
            if max_val is not None:
                var.assign(tf.minimum(var, max_val))

    def get_variable_values(self):
        """Get current values of all variables"""
        return {name: var.numpy() for name, var in self.variables.items()}

    def set_variable_values(self, values):
        """Set values for all variables"""
        for name, value in values.items():
            if name in self.variables:
                self.variables[name].assign(value)

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
        
        # Convert to numpy if needed
        def to_numpy(x):
            if isinstance(x, tf.Tensor):
                return x.numpy()
            return x
        
        # Convert tensors to numpy arrays and scale to 0-255
        current_image_np = to_numpy(current_image)
        input_image_np = to_numpy(input_image)
        
        # Calculate error
        error = np.abs(current_image_np - input_image_np)
        
        # Convert to uint8
        current_image_uint8 = (current_image_np * 255).astype(np.uint8)
        input_image_uint8 = (input_image_np * 255).astype(np.uint8)
        error_uint8 = (error * 255).astype(np.uint8)
        
        # Create visualization
        h, w = input_image_np.shape[:2]
        canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        # Original image
        canvas[:, :w] = input_image_uint8
        # Current approximation
        canvas[:, w:w*2] = current_image_uint8
        # Error visualization
        canvas[:, w*2:] = error_uint8
        
        # Add labels if requested
        if opts.label_snapshot:
            try:
                labels = ['Input', 'Current', 'Error']
                for i, label in enumerate(labels):
                    cv2.putText(canvas, label, (i*w + 10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (255, 255, 255), 2)
            except Exception as e:
                print(f"Warning: Failed to add labels: {e}")
        
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
        
        # Get gradients using model's trainable_variables property
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Clip gradients
        clipped_gradients = [tf.clip_by_norm(g, 1.0) if g is not None else None 
                           for g in gradients]
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(clipped_gradients, 
                                         self.model.trainable_variables))
        
        # Apply constraints
        self.model.apply_constraints()
        
        return loss, approx

class ProgressTracker:
    def __init__(self, total_iterations=None, time_limit=None, steps_per_iteration=1000):
        self.start_time = datetime.now()
        self.total_iterations = total_iterations
        self.time_limit = time_limit
        self.steps_per_iteration = steps_per_iteration
        self.best_loss = float('inf')
        self.last_improvement = 0
        self.iteration = 0
        
        try:
            from tqdm import tqdm
            self.progress_bar = tqdm(total=total_iterations if total_iterations else None,
                                   desc="Optimizing")
            self.use_progress_bar = True
        except ImportError:
            self.progress_bar = None
            self.use_progress_bar = False
            print("Note: Install 'tqdm' for progress bar support")
    
    def start_iteration(self):
        """Start a new iteration and return step range with optional progress bar"""
        self.iteration += 1
        
        if self.use_progress_bar:
            return tqdm(range(self.steps_per_iteration),
                       desc=f"Iteration {self.iteration}",
                       leave=False)  # Don't keep all iteration bars
        return range(self.steps_per_iteration)
    
    def update_loss(self, current_loss):
        """Update best loss tracking and return True if improved"""
        improved = False
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.last_improvement = self.iteration
            improved = True
            
        if self.use_progress_bar:
            self.progress_bar.set_postfix(
                loss=f"{current_loss:.6f}",
                best=f"{self.best_loss:.6f}"
            )
            self.progress_bar.update(1)
            
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

    def close(self):
        """Clean up progress bar"""
        if self.progress_bar:
            self.progress_bar.close()

def optimize_model(input_image, opts):
    """Main optimization function with improved progress tracking"""
    # Create model with proper image shape
    model = GaborModel('gabor', count=opts.num_gabors, 
                      image_shape=input_image.shape)
    
    # Create optimizer
    optimizer = GaborOptimizer(model, input_image, 
                             learning_rate=opts.learning_rate)
    
    # Initialize progress tracker
    progress = ProgressTracker(
        total_iterations=opts.total_iterations,
        time_limit=opts.time_limit,
        steps_per_iteration=opts.steps_per_iteration
    )
    
    print(f"\nStarting optimization with {opts.num_gabors} Gabors")
    print(f"Steps per iteration: {opts.steps_per_iteration}")
    
    try:
        while progress.should_continue():
            # Run optimization steps with progress bar
            for step in progress.start_iteration():
                loss, approx = optimizer.optimization_step()
                current_loss = loss.numpy()
                
                # Print detailed progress every 100 steps
                if step % 100 == 0:
                    elapsed = (datetime.now() - progress.start_time).total_seconds()
                    print(f"\rIteration {progress.iteration}, "
                          f"Step {step}/{opts.steps_per_iteration}, "
                          f"Loss: {current_loss:.6f}, "
                          f"Time: {format_time(elapsed)}", 
                          end="", flush=True)
            
            # Update tracking
            improved = progress.update_loss(current_loss)
            if improved:
                print(f"\n🌟 New best loss: {current_loss:.6f}")
                # Save best state
                best_state = model.get_variable_values()
            
            # Take snapshot if needed
            if opts.snapshot_prefix:
                try:
                    snapshot(model.generate_image(), input_image, opts, 
                            progress.iteration)
                except Exception as e:
                    print(f"\n⚠️  Warning: Failed to save snapshot: {e}")
                    # Save best model if requested
            if opts.save_best:
                save_model_state(model, opts.save_best)
                
            # Check for early stopping
            if (opts.early_stop and 
                progress.iteration - progress.last_improvement > opts.patience):
                print(f"\n⚡ Early stopping: No improvement for "
                      f"{progress.iteration - progress.last_improvement} iterations")
                break
            
            print()  # New line after iteration
    
    finally:
        # Clean up progress bar
        progress.close()
    
    # Restore best state
    if best_state is not None:
        model.set_variable_values(best_state)
    
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
    parser.add_argument('--patience', type=int, default=64,
                       help='Number of iterations without improvement before early stopping')
    parser.add_argument('--total-iterations', type=int, default=100,
                       help='Maximum number of iterations')
    parser.add_argument('--time-limit', type=float,
                       help='Time limit in seconds')
    parser.add_argument('--steps-per-iteration', type=int, default=256,
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
        image = imageio.imread(path).astype(np.float32) / 255.0
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

def setup_gpu(opts):
    """Configure GPU based on command line options"""
    if opts.force_cpu:
        print("Forcing CPU usage as requested.")
        return False
        
    try:
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("No GPU devices found. Running on CPU.")
            return False

        # Configure memory growth if requested (default: True)
        if opts.memory_growth:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
        # Enable mixed precision if requested (default: True)
        if opts.mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        
        print(f"GPU(s) configured successfully:")
        print(f"- Found {len(gpus)} GPU(s)")
        print(f"- Mixed precision: {'enabled' if opts.mixed_precision else 'disabled'}")
        print(f"- Memory growth: {'enabled' if opts.memory_growth else 'disabled'}")
        return True
        
    except Exception as e:
        print(f"Warning: GPU configuration failed: {e}")
        print("Falling back to CPU.")
        return False

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