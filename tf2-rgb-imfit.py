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

class State:
    def __init__(self, params, gabor, con_loss):
        self.params = params
        self.gabor = gabor
        self.con_loss = con_loss

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
        
        # Set up parameter ranges for clipping
        self.gmin = tf.constant(GABOR_RANGE[:,0], dtype=tf.float32)
        self.gmax = tf.constant(GABOR_RANGE[:,1], dtype=tf.float32)
        
        # Initialize parameters
        if params is not None:
            if not isinstance(params, tf.Variable):
                params = tf.Variable(params, trainable=True, dtype=tf.float32)
            self.params = params
        else:
            self.params = self.initialize_parameters(num_parallel, ensemble_size)
        
        # Learning rate parameters
        self.initial_lr = learning_rate
        self.min_lr = learning_rate * 0.01
        self.current_lr = learning_rate
        self.restart_period = 1000
        self.iteration = 0
        
        # Initialize optimizer
        self.opt = tf.keras.optimizers.Adam(
            learning_rate=self.current_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Training history
        self.loss_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Initial forward pass
        self._forward_pass()

    def initialize_parameters(self, num_parallel, ensemble_size):
        """Initialize Gabor parameters with improved initialization scheme"""
        init_params = np.zeros((num_parallel, GABOR_NUM_PARAMS, ensemble_size), dtype=np.float32)
        
        for i in range(GABOR_NUM_PARAMS):
            if GABOR_PARAM_H0 <= i <= GABOR_PARAM_H2:
                # Initialize heights with small positive values
                init_params[:,i,:] = np.random.uniform(0.01, 0.05, 
                                                     (num_parallel, ensemble_size))
            elif i in [GABOR_PARAM_L, GABOR_PARAM_T, GABOR_PARAM_S]:
                # Initialize scale parameters with reasonable values
                min_val = GABOR_RANGE[i, 0]
                max_val = GABOR_RANGE[i, 1]
                init_params[:,i,:] = np.random.uniform(
                    min_val + (max_val-min_val)*0.1,
                    min_val + (max_val-min_val)*0.5,
                    (num_parallel, ensemble_size)
                )
            else:
                # Initialize other parameters uniformly within their ranges
                init_params[:,i,:] = np.random.uniform(
                    GABOR_RANGE[i, 0],
                    GABOR_RANGE[i, 1],
                    (num_parallel, ensemble_size)
                )
        
        return tf.Variable(init_params, trainable=True, dtype=tf.float32)

    def _forward_pass(self):
            """Improved forward pass with better numerical stability"""
            with tf.name_scope('forward_pass'):
                # Reshape gmin and gmax for broadcasting
                gmin = tf.reshape(self.gmin, [1, GABOR_NUM_PARAMS, 1])
                gmax = tf.reshape(self.gmax, [1, GABOR_NUM_PARAMS, 1])
                
                # Clip parameters to valid ranges
                self.cparams = tf.clip_by_value(self.params, gmin, gmax)
                
                # Extract parameters
                u = self.cparams[:,GABOR_PARAM_U,:]
                v = self.cparams[:,GABOR_PARAM_V,:]
                r = self.cparams[:,GABOR_PARAM_R,:]
                l = self.cparams[:,GABOR_PARAM_L,:]
                t = self.cparams[:,GABOR_PARAM_T,:]
                s = self.cparams[:,GABOR_PARAM_S,:]
                
                # Extract RGB parameters
                h = self.cparams[:,GABOR_PARAM_H0:GABOR_PARAM_H0+3,:]
                p = self.cparams[:,GABOR_PARAM_P0:GABOR_PARAM_P0+3,:]
                
                # Add dimensions for broadcasting
                u = u[:,None,None,None,:]
                v = v[:,None,None,None,:]
                r = r[:,None,None,None,:]
                l = l[:,None,None,None,:]
                t = t[:,None,None,None,:]
                s = s[:,None,None,None,:]
                h = h[:,None,None,:,:]
                p = p[:,None,None,:,:]
                
                # Compute Gabor function with improved numerical stability
                cr = tf.cos(r)
                sr = tf.sin(r)
                f = tf.cast(2*np.pi, tf.float32) / tf.maximum(l, 1e-6)
                s2 = tf.maximum(s*s, 1e-6)
                t2 = tf.maximum(t*t, 1e-6)
                
                xp = self.x - u
                yp = self.y - v
                
                b1 = cr*xp + sr*yp
                b2 = -sr*xp + cr*yp
                
                b12 = b1*b1
                b22 = b2*b2
                
                # Prevent numerical instability in exponential
                exp_term = tf.clip_by_value(-b12/(2*s2) - b22/(2*t2), -88.0, 88.0)
                w = tf.exp(exp_term)
                
                k = f*b1 + p
                ck = tf.cos(k)
                
                # Combine components
                self.gabor = tf.identity(h * w * ck, name='gabor')
                self.approx = tf.reduce_sum(self.gabor, axis=4, name='approx')
                
                if self.target is not None:
                    self._compute_losses()
    def _compute_losses(self):
        """Improved loss computation with regularization"""
        # Compute reconstruction error
        self.err = tf.multiply((self.target - self.approx), self.weight)
        err_sqr = 0.5 * self.err**2
        
        # Add L2 regularization
        l2_reg = 0.001 * tf.reduce_sum(tf.square(self.params))
        
        # Per-fit error losses (average across h/w/c)
        self.err_loss_per_fit = tf.reduce_mean(err_sqr, axis=(1,2,3))
        
        # Overall error loss with regularization
        self.err_loss = tf.reduce_mean(self.err_loss_per_fit) + l2_reg
        
        # Compute constraints
        l = self.cparams[:,GABOR_PARAM_L,:]
        s = self.cparams[:,GABOR_PARAM_S,:]
        t = self.cparams[:,GABOR_PARAM_T,:]
        
        constraints = [
            tf.nn.softplus(-(s - l/32)),
            tf.nn.softplus(-(l/2 - s)),
            tf.nn.softplus(-(t - s)),
            tf.nn.softplus(-(8*s - t))
        ]
        
        # Stack constraints
        self.constraints = tf.stack(constraints, axis=2)
        
        # Compute constraint losses
        self.con_losses = tf.reduce_sum(self.constraints, axis=2)
        self.con_loss_per_fit = tf.reduce_sum(self.con_losses, axis=1)
        self.con_loss = tf.reduce_mean(self.con_loss_per_fit)
        
        # Total loss
        self.loss_per_fit = self.err_loss_per_fit + self.con_loss_per_fit
        self.loss = self.err_loss + self.con_loss

    @tf.function
    def train_step(self):
        """Improved training step with gradient handling"""
        with tf.GradientTape() as tape:
            self._forward_pass()
            loss = self.loss
        
        # Compute gradients
        gradients = tape.gradient(loss, [self.params])
        
        # Gradient clipping and normalization
        gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
        
        # Check for valid gradients
        if gradients[0] is not None:
            # Normalize gradients
            grad_norm = tf.linalg.global_norm(gradients)
            if grad_norm > 0:
                gradients = [g / (grad_norm + 1e-7) for g in gradients]
            
            # Apply gradients
            self.opt.apply_gradients(zip(gradients, [self.params]))
            
            # Update learning rate
            self._update_learning_rate()
        
        return {
            'loss': loss,
            'gabor': self.gabor,
            'approx': self.approx,
            'params': self.params,
            'err_loss_per_fit': self.err_loss_per_fit,
            'con_loss_per_fit': self.con_loss_per_fit
        }
    def _update_learning_rate(self):
        """Cosine decay with warm restarts"""
        self.iteration += 1
        
        # Compute current position in restart cycle
        current_iter = self.iteration % self.restart_period
        
        # Cosine decay with warm restarts
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * current_iter / self.restart_period))
        new_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        
        # Update optimizer learning rate
        self.current_lr = new_lr
        self.opt.learning_rate.assign(new_lr)

    def reset_optimization(self):
        """Reset optimization state"""
        self.iteration = 0
        self.current_lr = self.initial_lr
        self.opt.learning_rate.assign(self.initial_lr)
        self.loss_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0

    def get_current_state(self):
        """Get current model state"""
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
    
def evaluate_fit_quality(target, approx, weight=None):
    """Evaluate the quality of the fit with multiple metrics"""
    if weight is None:
        weight = np.ones_like(target)
        
    # Compute weighted MSE
    mse = np.mean(weight * (target - approx)**2)
    
    # Compute PSNR
    max_val = target.max() - target.min()
    psnr = 20 * np.log10(max_val) - 10 * np.log10(max(mse, 1e-10))
    
    # Compute weighted correlation coefficient
    weighted_target = weight * target
    weighted_approx = weight * approx
    correlation = np.corrcoef(weighted_target.flatten(), weighted_approx.flatten())[0,1]
    
    return {
        'mse': float(mse),
        'psnr': float(psnr),
        'correlation': float(correlation)
    }
def setup_models(opts, inputs, state):
    """Set up the models for optimization"""
    # Validate dimensions
    if opts.num_models <= 0:
        raise ValueError("num_models must be positive")
    if opts.num_local <= 0:
        raise ValueError("num_local must be positive")
        
    weight_tensor = tf.constant(inputs.weight_image)
    x_tensor = tf.constant(inputs.x.reshape(1,1,-1,1,1))
    y_tensor = tf.constant(inputs.y.reshape(1,-1,1,1,1))

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
        preview_shape = scale_shape(inputs.target_tensor.shape[:2].as_list(),
                                  opts.preview_size)
        _, x_preview, y_preview = normalized_grid(preview_shape)
        
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
    state = State(
        params=params,
        gabor=np.zeros(inputs.input_image.shape + (opts.num_models,), dtype=np.float32),
        con_loss=np.zeros(opts.num_models, dtype=np.float32)
)
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

    # Rescale images for display:
    # We assume that both the target and the approximation are in [-1, 1].
    approx_img   = rescale(cur_approx, -1, 1)

    # Create a montage of the images.
    # Montage order: Target | Approximation | Residual | Absolute Error
    out_img = approx_img
    
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
    print(f"\nStarting local optimization for model {current_model + 1}")
    initial_loss = None
    
    for i in range(opts.local_iter):
        try:
            result = models.local.train_step()
            current_loss = float(result['loss'])
            
            if initial_loss is None:
                initial_loss = current_loss
                
            if i % 5 == 0:
                print(f"  Step {i}: loss = {current_loss:.6f}")
                
        except Exception as e:
            print(f"Error in local optimization step {i}: {e}")
            break
    
    if initial_loss is not None:
        improvement = initial_loss - current_loss
        print(f"Local optimization complete: loss improved by {improvement:.6f}")
        
    return current_loss

######################################################################

def main():
    # Set up command-line options, image inputs, and models
    opts = get_options()
    inputs = setup_inputs(opts)
    state = setup_state(opts, inputs)
    models = setup_models(opts, inputs, state)

    # Initialize tracking variables
    loop_count = 0
    best_loss = float('inf')
    # If initial parameter files were provided, load them
    prev_best_loss, initial_filter_count = load_params(opts, inputs, models, state)
    
    # Starting filter count is the number already in the state
    current_model = initial_filter_count
    start_time = datetime.now()
    
    try:
        while True:
            # Check termination conditions
            if opts.time_limit is not None:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > opts.time_limit:
                    print('Time limit reached!')
                    break
                    
            if opts.total_iterations is not None and loop_count >= opts.total_iterations:
                print(f'Iteration limit ({opts.total_iterations}) reached!')
                break

            print(f"\nIteration #{loop_count+1}:")

            # 1. Compute residual
            if current_model > 0:
                current_approx = tf.reduce_sum(state.gabor[:,:,:,:current_model], axis=3)
                residual = inputs.input_image - current_approx.numpy()
            else:
                residual = inputs.input_image

            # Update target for local optimization
            inputs.target_tensor.assign(residual)

            # 2. Local optimization phase
            print(f"Performing local optimization for model {current_model+1}/{opts.num_models}")
            
            # Reset local model for new optimization
            models.local.reset_optimization()
            
            best_local_loss = float('inf')
            best_local_params = None
            
            # Run multiple local optimization attempts
            for attempt in range(opts.local_iter):
                result = models.local.train_step()
                current_loss = float(result['loss'])
                
                if current_loss < best_local_loss:
                    best_local_loss = current_loss
                    best_local_params = result['params'].numpy()
                
                if attempt % 5 == 0:
                    print(f"  Local step {attempt}: loss = {current_loss:.6f}")

            # Update state with best local result
            if best_local_params is not None:
                state.params[:, current_model] = best_local_params[0,:,0]
                current_model = min(current_model + 1, opts.num_models - 1)

            # 3. Periodic full optimization
            if (loop_count + 1) % opts.full_every == 0:
                print("\nPerforming full joint optimization...")
                
                # Reset full model for optimization
                models.full.reset_optimization()
                
                for i in range(opts.full_iter):
                    result = models.full.train_step()
                    current_loss = float(result['loss'])
                    
                    if current_loss < best_loss:
                        best_loss = current_loss
                        # Update state with improved parameters
                        state.params = result['params'].numpy()[0]
                        
                        # Save intermediate result
                        if opts.output is not None:
                            np.savetxt(opts.output, state.params.transpose(), 
                                     fmt='%f', delimiter=',')
                    
                    if i % 50 == 0:
                        print(f"  Full step {i}: loss = {current_loss:.6f}")

                # Update preview and create snapshot
                if opts.preview_size:
                    models.preview.params.assign(models.full.params)
                    _ = models.preview._forward_pass()
                
                snapshot(result['gabor'].numpy()[0], 
                        result['approx'].numpy()[0],
                        opts, inputs, models,
                        loop_count, 'full')

            loop_count += 1

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user!")
    finally:
        # Save final results
        if opts.output is not None:
            print(f"\nSaving final parameters to {opts.output}")
            np.savetxt(opts.output, state.params.transpose(), fmt='%f', delimiter=',')
            
        # Create final snapshot
        final_result = models.full.get_current_state()
        snapshot(final_result['gabor'].numpy()[0],
                final_result['approx'].numpy()[0],
                opts, inputs, models,
                loop_count, 'final')
        
        print("\nOptimization completed!")
        print(f"Final loss: {float(final_result['loss']):.6f}")
        print(f"Total iterations: {loop_count}")
        print(f"Total time: {(datetime.now() - start_time).total_seconds():.1f}s")

def evaluate_fit_quality(target, approx, weight=None):
    """Evaluate the quality of the fit with multiple metrics"""
    if weight is None:
        weight = np.ones_like(target)
        
    # Compute weighted MSE
    mse = np.mean(weight * (target - approx)**2)
    
    # Compute PSNR
    max_val = target.max() - target.min()
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    
    # Compute weighted correlation coefficient
    weighted_target = weight * target
    weighted_approx = weight * approx
    correlation = np.corrcoef(weighted_target.flatten(), weighted_approx.flatten())[0,1]
    
    return {
        'mse': mse,
        'psnr': psnr,
        'correlation': correlation
    }

def save_visualization(output_path, original, approximation, residual):
    """Save a visualization of the fitting results"""
    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    
    # Plot approximation
    ax2.imshow(approximation)
    ax2.set_title('Gabor Approximation')
    ax2.axis('off')
    
    # Plot residual
    residual_plot = ax3.imshow(residual, cmap='RdBu')
    ax3.set_title('Residual')
    ax3.axis('off')
    plt.colorbar(residual_plot, ax=ax3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Enable memory growth for GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Run main optimization
    main()
