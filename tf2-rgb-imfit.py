from __future__ import print_function
import re, os, sys, argparse
from datetime import datetime
from collections import namedtuple
import tensorflow as tf
import numpy as np
from PIL import Image
import traceback
import json
import copy
from typing import Dict, Tuple, Optional
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x
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


def format_time(seconds):
    """Format time in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"

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

    def update_constraints(self, max_sigma=None, max_frequency=None):
        """Update model constraints for curriculum learning"""
        if max_sigma is not None:
            self.variables['sigma'].assign(
                tf.clip_by_value(self.variables['sigma'], 1.0, max_sigma))
        
        if max_frequency is not None:
            self.variables['frequency'].assign(
                tf.clip_by_value(self.variables['frequency'], 0.02, max_frequency))

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

def snapshot(current_image, input_image, opts, iteration, extra_info=None):
    """Save a snapshot of the current optimization state"""
    if not opts.snapshot_prefix:
        return
    
    try:
        # Create output directory if it doesn't exist
        outdir = os.path.dirname(opts.snapshot_prefix)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        # Generate output filename with stage info if provided
        if extra_info:
            outfile = f'{opts.snapshot_prefix}-{int(iteration):06d}-{extra_info}.png'
        else:
            outfile = f'{opts.snapshot_prefix}-{int(iteration):06d}.png'
        
        # Ensure we have numpy arrays
        def to_numpy(x):
            if isinstance(x, tf.Tensor):
                return x.numpy()
            if isinstance(x, np.ndarray):
                return x
            raise ValueError(f"Unsupported type for image: {type(x)}")
        
        try:
            current_image_np = to_numpy(current_image)
            input_image_np = to_numpy(input_image)
        except Exception as e:
            print(f"Error converting images to numpy: {e}")
            return
        
        # Ensure proper value range [0, 1]
        current_image_np = np.clip(current_image_np, 0, 1)
        input_image_np = np.clip(input_image_np, 0, 1)
        
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
            print(f"Saved snapshot to {outfile}")
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
    def __init__(self, model, input_image, learning_rate=0.01, weights=None):
        self.model = model
        self.input_image = input_image
        self.weights = weights
        if weights is not None:
            self.weights = weights[..., np.newaxis]
        
        # Use a more sophisticated learning rate schedule
        self.initial_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,  # Momentum term
            beta_2=0.999,  # RMSprop term
            epsilon=1e-7,  # Numerical stability
            amsgrad=True  # Use AMSGrad variant
        )
        
        self.best_loss = float('inf')
        self.best_state = None
        self.loss_history = []
        self.plateau_threshold = 1e-6

    @tf.function
    def optimization_step(self):
        """Improved optimization step with gradient accumulation"""
        with tf.GradientTape() as tape:
            # Generate current approximation
            approx = self.model.generate_image()
            
            # Calculate loss with regularization
            if self.weights is not None:
                diff = approx - self.input_image
                weighted_diff = diff * self.weights
                reconstruction_loss = tf.reduce_mean(tf.square(weighted_diff))
            else:
                reconstruction_loss = tf.reduce_mean(tf.square(approx - self.input_image))
            
            # Add regularization to prevent extreme values
            l2_reg = tf.reduce_mean(tf.square(self.model.amplitude)) * 0.001
            frequency_reg = tf.reduce_mean(tf.square(self.model.frequency)) * 0.001
            
            total_loss = reconstruction_loss + l2_reg + frequency_reg
            
            # Create components dict for monitoring
            components = {
                'reconstruction': reconstruction_loss,
                'l2': l2_reg,
                'frequency': frequency_reg
            }
        
        # Get and process gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Gradient clipping with dynamic norm
        global_norm = tf.linalg.global_norm(gradients)
        clip_norm = tf.maximum(1.0, global_norm / 100.0)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
        
        # Apply processed gradients
        self.optimizer.apply_gradients(zip(clipped_gradients, 
                                         self.model.trainable_variables))
        
        # Apply constraints with smoothing
        self.model.apply_constraints()
        
        return total_loss, components, approx

    def save_current_state(self, filename):
        """Save current model state"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Get current state
            state = self.model.get_variable_values()
            
            # Save state
            np.savez(filename, 
                     **state,
                     loss=self.best_loss,
                     learning_rate=self.optimizer.learning_rate.numpy())
            
            print(f"Saved optimizer state to {filename}")
            return True
        except Exception as e:
            print(f"Failed to save optimizer state: {e}")
            return False

    def prepare_for_stage(self, stage: int, max_sigma: float, max_frequency: float):
        """Prepare optimizer for new curriculum stage"""
        # Reset optimizer state while keeping best model
        self.optimizer.learning_rate.assign(self.initial_learning_rate)
        self.steps_without_improvement = 0
        self.loss_history = []
        
        # Update model constraints
        self.model.update_constraints(max_sigma=max_sigma, max_frequency=max_frequency)
        
        # Restore best state if available
        if self.best_state is not None:
            self.model.set_variable_values(self.best_state)
            print(f"Restored best model state for stage {stage}")

class StateManager:
    """Manages optimization state and recovery"""
    def __init__(self, model, optimizer, save_dir):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.best_state = None
        self.best_loss = float('inf')
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    def save_checkpoint(self, stage, step, loss):
        """Save optimization checkpoint"""
        if not self.save_dir:
            return
        
        checkpoint = {
            'model_state': self.model.get_variable_values(),
            'optimizer_learning_rate': self.optimizer.current_learning_rate,
            'stage': stage,
            'step': step,
            'loss': loss,
            'best_loss': self.best_loss
        }
        
        checkpoint_path = os.path.join(
            self.save_dir, f'checkpoint_stage{stage}_step{step}.npz')
        
        try:
            np.savez_compressed(checkpoint_path, **checkpoint)
            print(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load optimization checkpoint"""
        try:
            checkpoint = np.load(checkpoint_path, allow_pickle=True)
            
            # Restore model state
            self.model.set_variable_values(checkpoint['model_state'].item())
            
            # Restore optimizer learning rate
            self.optimizer.current_learning_rate = float(checkpoint['optimizer_learning_rate'])
            self.optimizer.optimizer.learning_rate.assign(self.optimizer.current_learning_rate)
            
            # Restore other state variables
            self.best_loss = float(checkpoint['best_loss'])
            
            print(f"Loaded checkpoint from {checkpoint_path}")
            return (int(checkpoint['stage']), 
                   int(checkpoint['step']),
                   float(checkpoint['loss']))
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None
    
    def update_best_state(self, loss):
        """Update best state if current loss is better"""
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_state = self.model.get_variable_values()
            return True
        return False
    
    def restore_best_state(self):
        """Restore model to best known state"""
        if self.best_state is not None:
            self.model.set_variable_values(self.best_state)
            print(f"Restored best model state (loss: {self.best_loss:.6f})")
            return True
        return False

def optimize_with_curriculum(input_image, opts, weights=None):
    """Progressive optimization with curriculum learning"""
    print("\nInitializing optimization...")
    
    # Setup GPU if available
    using_gpu = setup_gpu_memory()
    
    try:
        # Create model and optimizer
        model = GaborModel('gabor', count=opts.num_gabors, 
                         image_shape=input_image.shape)
        optimizer = GaborOptimizer(model, input_image, 
                                 learning_rate=opts.learning_rate,
                                 weights=weights)
        
        # Create state manager
        state_manager = StateManager(
            model, optimizer, 
            save_dir=os.path.join(opts.output_dir, 'checkpoints'))
        
        # Load initial state if provided
        if opts.load_state and os.path.exists(opts.load_state):
            state_manager.load_checkpoint(opts.load_state)
        
        # Curriculum learning stages
        sigma_schedules = [20.0, 10.0, 5.0, 2.0]
        frequency_schedules = [0.02, 0.05, 0.1, 0.2]
        
        total_steps = 0
        start_time = datetime.now()
        
        try:
            for stage, (max_sigma, max_freq) in enumerate(zip(
                sigma_schedules, frequency_schedules)):
                
                print(f"\nStage {stage + 1}/{len(sigma_schedules)}")
                print(f"Max sigma: {max_sigma:.1f}, Max frequency: {max_freq:.3f}")
                
                # Prepare optimizer for new stage
                optimizer.prepare_for_stage(stage + 1, max_sigma, max_freq)
                
                # Calculate iterations for this stage
                stage_iterations = opts.total_iterations // len(sigma_schedules)
                
                with tqdm(total=stage_iterations, 
                         desc=f"Stage {stage + 1}") as pbar:
                    
                    for step in range(stage_iterations):
                        try:
                            # Check time limit
                            if opts.time_limit is not None:
                                elapsed = (datetime.now() - start_time).total_seconds()
                                if elapsed > opts.time_limit:
                                    print("\nTime limit reached")
                                    return model, state_manager.best_loss
                            
                            # Optimization step
                            loss, components, image = optimizer.optimization_step()
                            total_steps += 1
                            
                            # Update state manager
                            if state_manager.update_best_state(loss):
                                state_manager.save_checkpoint(
                                    stage, step, loss)
                            
                            # Update progress with more detailed loss information
                            if step % 10 == 0:
                                pbar.set_postfix(
                                    loss=f"{loss:.6f}",
                                    recon=f"{components['reconstruction']:.6f}"
                                )
                                pbar.update(10)
                            
                            # Save snapshot
                            if opts.snapshot_prefix and step % opts.snapshot_frequency == 0:
                                snapshot(image, input_image, opts, 
                                        total_steps, f"stage_{stage+1}")
                            
                            # Early stopping check
                            if optimizer.should_stop_early(loss):
                                print("\nEarly stopping triggered")
                                break
                            
                            # Learning rate adjustment
                            optimizer.adjust_learning_rate(loss)
                            
                        except tf.errors.ResourceExhaustedError:
                            print("\nOOM error - attempting recovery...")
                            cleanup_gpu_memory()
                            state_manager.restore_best_state()
                            continue
                        
                        except Exception as e:
                            print(f"\nError during optimization: {e}")
                            traceback.print_exc()
                            state_manager.restore_best_state()
                            continue
        
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user")
            state_manager.restore_best_state()
        
        finally:
            # Final cleanup
            if using_gpu:
                cleanup_gpu_memory()
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"\nOptimization completed in {format_time(elapsed_time)}")
        print(f"Final loss: {state_manager.best_loss:.6f}")
        
        return model, state_manager.best_loss
        
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        if using_gpu:
            cleanup_gpu_memory()
        raise

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
    """Create and setup argument parser with all relevant arguments"""
    parser = argparse.ArgumentParser(description='TF2 Gabor Image Fitting')
    
    # Input/Output arguments
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--input', type=str, required=True,
                         help='Input image path')
    io_group.add_argument('--weights', type=str,
                         help='Optional weight image path (greyscale)')
    io_group.add_argument('--output-dir', type=str, default='results',
                         help='Directory for output files')
    io_group.add_argument('--snapshot-prefix', type=str,
                         help='Prefix for snapshot filenames (e.g., results/output)')
    io_group.add_argument('--save-best', type=str,
                         help='Save best model state to file')
    io_group.add_argument('--load-state', type=str,
                         help='Load initial model state from file')
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--num-gabors', type=int, default=100,
                           help='Number of Gabor functions')
    model_group.add_argument('--regularization', type=float, default=0.001,
                           help='L2 regularization strength')
    
    # Optimization arguments
    optim_group = parser.add_argument_group('Optimization')
    optim_group.add_argument('--learning-rate', type=float, default=0.01,
                            help='Initial learning rate')
    optim_group.add_argument('--steps-per-iteration', type=int, default=1000,
                            help='Optimization steps per iteration')
    optim_group.add_argument('--total-iterations', type=int, default=None,
                            help='Maximum number of iterations (None for unlimited)')
    optim_group.add_argument('--time-limit', type=float, default=None,
                            help='Time limit in seconds (None for unlimited)')
    optim_group.add_argument('--early-stop', action='store_true',
                            help='Enable early stopping')
    optim_group.add_argument('--patience', type=int, default=1000,
                            help='Patience for early stopping')
    optim_group.add_argument('--batch-size', type=int, default=4,
                            help='Batch size for optimization')
    
    # Annealing arguments
    anneal_group = parser.add_argument_group('Learning Rate Annealing')
    anneal_group.add_argument('--anneal-learning-rate', action='store_true',
                             help='Enable learning rate annealing')
    anneal_group.add_argument('--anneal-factor', type=float, default=0.7,
                             help='Factor to reduce learning rate by during annealing')
    anneal_group.add_argument('--anneal-patience', type=int, default=500,
                             help='Iterations before annealing')
    anneal_group.add_argument('--min-learning-rate', type=float, default=1e-6,
                             help='Minimum learning rate for annealing')
    
    # Visualization arguments
    vis_group = parser.add_argument_group('Visualization')
    vis_group.add_argument('--label-snapshot', action='store_true',
                          help='Add labels to snapshot images')
    vis_group.add_argument('--snapshot-frequency', type=int, default=1,
                          help='Save snapshot every N iterations')
    vis_group.add_argument('--progress-frequency', type=int, default=100,
                          help='Show progress every N steps')
    
    # Hardware configuration
    hw_group = parser.add_argument_group('Hardware')
    hw_group.add_argument('--force-cpu', action='store_true',
                         help='Force CPU usage even if GPU is available')
    hw_group.add_argument('--mixed-precision', action='store_true',
                         help='Enable mixed precision training')
    hw_group.add_argument('--memory-growth', action='store_true',
                         help='Enable GPU memory growth')
    
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

def load_input_image(path, weight_path=None, scale=1.0):
    """Load and preprocess input image and optional weights with scaling"""
    try:
        # Load main image and normalize to [0, 1]
        image = imageio.imread(path).astype(np.float32) / 255.0
        
        # Scale image if requested
        if scale != 1.0:
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            image = tf.image.resize(image, (new_h, new_w)).numpy()
        
        # Ensure 3 channels (RGB)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:  # RGBA
            image = image[..., :3]
        
        # Load and scale weights if provided
        weights = None
        if weight_path:
            try:
                weights = imageio.imread(weight_path).astype(np.float32) / 255.0
                if len(weights.shape) > 2:  # Convert to greyscale if needed
                    weights = np.mean(weights, axis=-1)
                
                # Scale weights to match image
                if scale != 1.0:
                    weights = tf.image.resize(weights[..., None], 
                                           (new_h, new_w))[..., 0].numpy()
                
                print(f"Loaded weight image from {weight_path}")
                print(f"Weight range: {weights.min():.3f} to {weights.max():.3f}")
                
            except Exception as e:
                print(f"Warning: Failed to load weights, using uniform weights: {e}")
                weights = None
        
        return image, weights
        
    except Exception as e:
        raise RuntimeError(f"Failed to load image {path}: {e}")

def setup_gpu():
    """Configure GPU and return True if GPU is available and configured"""
    try:
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("No GPU devices found. Running on CPU.")
            return False

        # Configure memory growth to prevent TF from taking all GPU memory
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Warning: Could not set memory growth for {gpu}: {e}")
        
        # Enable mixed precision for better GPU performance
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled")
        except Exception as e:
            print(f"Warning: Could not enable mixed precision: {e}")
        
        print(f"GPU(s) configured successfully:")
        print(f"- Found {len(gpus)} GPU(s)")
        print(f"- Memory growth enabled")
        return True
        
    except Exception as e:
        print(f"Warning: GPU configuration failed: {e}")
        print("Falling back to CPU.")
        return False

def save_model_state(model, filename):
    """Save model variables to text file in original format"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Get model state and flatten variables into a single array
        state = model.get_variable_values()
        flattened = []
        for name in ['pos_x', 'pos_y', 'sigma', 'theta', 'frequency', 
                    'phase', 'colors', 'amplitude', 'scale']:
            value = state[name]
            if len(value.shape) > 1:  # For colors array
                value = value.reshape(-1)
            flattened.extend(value)
        
        # Convert to numpy array and save
        params = np.array(flattened)
        np.savetxt(filename, params)
        print(f"✓ Saved model state to {filename}")
        
    except Exception as e:
        print(f"⚠️ Failed to save model state: {e}")
        traceback.print_exc()

def load_model_state(model, filename):
    """Load model variables from text file in original format"""
    try:
        # Load the flattened parameters
        params = np.loadtxt(filename)
        
        # Calculate sizes for each variable
        num_gabors = model.count
        sizes = {
            'pos_x': num_gabors,
            'pos_y': num_gabors,
            'sigma': num_gabors,
            'theta': num_gabors,
            'frequency': num_gabors,
            'phase': num_gabors,
            'colors': num_gabors * 3,  # RGB values
            'amplitude': num_gabors,
            'scale': num_gabors
        }
        
        # Split the parameters into variables
        state = {}
        start = 0
        for name, size in sizes.items():
            end = start + size
            value = params[start:end]
            
            # Reshape colors back to 2D
            if name == 'colors':
                value = value.reshape(num_gabors, 3)
                
            state[name] = value
            start = end
        
        # Verify we used all parameters
        if start != len(params):
            raise ValueError(f"Parameter count mismatch: expected {start}, got {len(params)}")
        
        # Set the variables
        model.set_variable_values(state)
        print(f"✓ Loaded model state from {filename}")
        
    except Exception as e:
        print(f"⚠️ Failed to load model state: {e}")
        traceback.print_exc()
        raise

def setup_gpu_memory():
    """Configure GPU memory growth to prevent OOM errors"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("No GPU devices found. Running on CPU.")
            return False

        # Enable memory growth for all GPUs
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Enabled memory growth for GPU: {gpu}")
            except RuntimeError as e:
                print(f"Error setting memory growth for {gpu}: {e}")
                continue

        # Set mixed precision policy
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Enabled mixed precision training")
        except Exception as e:
            print(f"Warning: Could not enable mixed precision: {e}")

        return True
    except Exception as e:
        print(f"GPU setup failed: {e}")
        return False

def cleanup_gpu_memory():
    """Clean up GPU memory after optimization"""
    try:
        tf.keras.backend.clear_session()
        print("Cleaned up GPU memory")
    except Exception as e:
        print(f"Warning: Memory cleanup failed: {e}")

def main():
    parser = setup_argument_parser()
    opts = parser.parse_args()
    
    # Configure GPU
    using_gpu = setup_gpu()
    
    try:
        # Load and preprocess input
        input_image, weights = load_input_image(opts.input, opts.weights)
        
        # Run optimization with curriculum learning
        model, final_loss = optimize_with_curriculum(
            input_image, opts, weights=weights)
        
        # Save final model
        if opts.save_best:
            save_model_state(model, opts.save_best)
        
        print(f"\nOptimization complete!")
        print(f"Final loss: {final_loss:.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())