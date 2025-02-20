import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.signal
from torchvision.transforms.functional import gaussian_blur

GABOR_MIN ={
    'u': 0,
    'v': 0,
    'theta': 0,
    'rel_sigma': 0.05,
    'rel_freq': 0.03,
    'gamma': 0.03,
    'psi': -3,
    'amplitude': 0.00
}

GABOR_MAX ={
    'u': 1,
    'v': 1,
    'theta': 2,
    'rel_sigma': 0.7,
    'rel_freq': 5,
    'gamma': 0.15,
    'psi': 3,
    'amplitude': 0.2
}

class GaborLayer(nn.Module):
    def __init__(self, num_gabors=256, base_scale=64):
        super().__init__()
        self.base_scale = base_scale
        
        # Initialize BatchNorm2d with the correct number of channels
        #self.batch_norm = nn.BatchNorm2d(num_gabors)  # This should match the number of output channels
        
        # Initialize parameters with conservative ranges
        self.u = nn.Parameter(torch.rand(num_gabors).normal_(GABOR_MIN['u'], GABOR_MAX['u']))  
        self.v = nn.Parameter(torch.rand(num_gabors).normal_(GABOR_MIN['v'], GABOR_MAX['v']))  
        self.theta = nn.Parameter(torch.rand(num_gabors).normal_(GABOR_MIN['theta'], GABOR_MAX['theta']))  
        self.rel_sigma = nn.Parameter(torch.randn(num_gabors).normal_(GABOR_MIN['rel_sigma'], GABOR_MAX['rel_sigma']))  
        self.rel_freq = nn.Parameter(torch.randn(num_gabors).normal_(GABOR_MIN['rel_freq'], GABOR_MAX['rel_freq']))   
        self.gamma = nn.Parameter(torch.zeros(num_gabors).normal_(GABOR_MIN['gamma'], GABOR_MAX['gamma']))  
        self.psi = nn.Parameter(torch.rand(num_gabors, 3).normal_(GABOR_MIN['psi'], GABOR_MAX['psi']))  
        self.amplitude = nn.Parameter(torch.randn(num_gabors, 3).normal_(GABOR_MIN['amplitude'], GABOR_MAX['amplitude']))  
        
        self.dropout = nn.Dropout(p=0.01)

    def load_state_dict(self, state_dict, strict=True):
        with torch.no_grad():
            state_dict['u']
            state_dict['v']
            state_dict['theta']                                                                                                                                                  
            state_dict['rel_sigma']
            state_dict['rel_freq']
            state_dict['psi']
            state_dict['gamma']
            state_dict['amplitude']
        
        return super().load_state_dict(state_dict, strict)

    def forward(self, grid_x, grid_y, temperature=1.0, dropout_active=True):
        H, W = grid_x.shape
        image_size = max(H, W)
        base_size = float(image_size)/ float(self.base_scale)

        self.enforce_parameter_ranges()
        
        # Safe parameter transformations with gradient preservation
        u = self.u.clamp(-1, 1)
        v = self.v.clamp(-1, 1)
        theta = self.theta.clamp(-2, 2)*2*np.pi
        
        # Ensure positive sigma with safe scaling
        sigma = self.rel_sigma.clamp(1e-5, 5)
        
        # Safe aspect ratio
        gamma = self.gamma.clamp(1e-5, 5)
        cr = torch.cos(theta[:,None,None])
        sr = torch.sin(theta[:,None,None])
        
        # Compute rotated coordinates
        x_rot = (grid_x[None,:,:] - u[:,None,None]) * cr + \
                (grid_y[None,:,:] - v[:,None,None]) * sr
        y_rot = -(grid_x[None,:,:] - u[:,None,None]) * sr + \
                (grid_y[None,:,:] - v[:,None,None]) * cr

        # Safe gaussian computation

        gaussian = torch.exp(torch.clamp(
            -(x_rot**2)/(2*(sigma[:,None,None]**2)) - (y_rot**2)/(2*(gamma[:,None,None]**2)),
            min=-80, max=80
        ))
        
        # Safe sinusoid computation with frequency scaling
        freq = np.float32(2*np.pi) / torch.exp(self.rel_freq)
        phase = self.psi*2*np.pi
        sinusoid = torch.cos(freq[:,None,None,None] * x_rot[:, None, :, :] + 
                           phase[:, :, None, None])
        
        # Combine components safely
        gabors = self.amplitude[:, :, None, None] * gaussian[:, None, :, :] * sinusoid
        result = torch.sum(gabors, dim=0)  # This should be [num_gabors, height, width]
        
        # Ensure result is 4D before batch normalization
        #result = result.unsqueeze(0)  # Add batch dimension, now [1, num_gabors, height, width]
        
        # Apply batch normalization
        #result = self.batch_norm(result)  # Apply batch normalization
        result = torch.clamp(result, -1, 1)  # Clamp to normalized range       
        return result  # Remove batch dimension for output

    def enforce_parameter_ranges(self):
        """Enforce valid parameter ranges"""
        with torch.no_grad():
            self.u.clamp_(-1, 1)
            self.v.clamp_(-1, 1)
            self.theta.clamp_(-2, 2)
            self.rel_sigma.clamp_(1e-5,5)
            # self.rel_freq.clamp_(1e-5,5)
            self.psi.clamp_(-1, 1)
            self.gamma.clamp_(1e-5,5)
            self.amplitude.clamp_(0,1)

class ImageFitter:
    def __init__(self, image_path, weight_path=None, num_gabors=256, target_size=None, 
                 device='cuda' if torch.cuda.is_available() else 'cpu', init=None,
                 global_lr=0.03, local_lr=0.01, init_size=128, mutation_strength=0.01):  # Add learning rate parameters
        image = Image.open(image_path).convert('RGB')
        
        # Resize image if target_size is specified
        if target_size is not None:
            image = self.resize_image(image, target_size)
        
        # Enhanced preprocessing pipeline
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.target = transform(image).to(device)
        h, w = self.target.shape[-2:]

       #load weights if provided
        if weight_path:
            weight_img = Image.open(weight_path).convert('L')  # Convert to grayscale
            weight_img = weight_img.resize((w, h), Image.Resampling.LANCZOS)
            self.weights = transforms.ToTensor()(weight_img).to(device)
            # Normalize weights to average to 1
            self.weights = self.weights / self.weights.mean()
        else:
            self.weights = torch.ones((h, w), device=device)

        # Create coordinate grid
        y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
        self.grid_x = x.to(device)
        self.grid_y = y.to(device)
        
        # Initialize model
        self.model = GaborLayer(num_gabors,init_size).to(device)
    
        # Initialize model parameters if provided
        if init:
            self.init_parameters(init)
        
        # Initialize optimizers with provided learning rates
        self.global_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=global_lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        self.local_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=local_lr,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
      
        self.optimizer = self.global_optimizer  # Start with global optimizer
        
        # Initialize schedulers for both phases
        self.global_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.global_optimizer,
            mode='min',
            factor=0.5,
            patience=50,
            verbose=True,
            min_lr=1e-5
        )
        
        self.local_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.local_optimizer,
            mode='min',
            factor=0.5,
            patience=30,
            verbose=True,
            min_lr=1e-6
        )
        
        self.scheduler = self.global_scheduler  # Start with global scheduler
        
        # Initialize best loss tracking / needs to be implemented
        self.best_loss = float('inf')
        self.best_state = None
        
        # Add temperature scheduling
        self.initial_temp = 0.1
        self.min_temp = 0.00000001
        self.current_temp = self.initial_temp
        
        # Add mutation probability
        self.mutation_prob = 0.01
        self.mutation_strength = mutation_strength
        # Add phase tracking
        self.optimization_phase = 'global'  # 'global' or 'local'
        self.phase_split = 0.5  # default value, will be updated from args
    
    def add_gabor(self):
        """Add a new Gabor filter to the model."""
        new_gabor = GaborLayer(num_gabors=1, base_scale=self.model.base_scale).to(self.model.u.device)
        self.model.u = nn.Parameter(torch.cat((self.model.u, new_gabor.u)))
        self.model.v = nn.Parameter(torch.cat((self.model.v, new_gabor.v)))
        self.model.theta = nn.Parameter(torch.cat((self.model.theta, new_gabor.theta)))
        self.model.rel_sigma = nn.Parameter(torch.cat((self.model.rel_sigma, new_gabor.rel_sigma)))
        self.model.rel_freq = nn.Parameter(torch.cat((self.model.rel_freq, new_gabor.rel_freq)))
        self.model.gamma = nn.Parameter(torch.cat((self.model.gamma, new_gabor.gamma)))
        self.model.psi = nn.Parameter(torch.cat((self.model.psi, new_gabor.psi)))
        self.model.amplitude = nn.Parameter(torch.cat((self.model.amplitude, new_gabor.amplitude)))
        self.global_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.global_lr,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        self.local_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.local_lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        self.optimizer = self.global_optimizer
    def init_parameters(self, init):
        """Initialize parameters from a saved model"""
        if init:
            self.load_model(init)
            print("Initialized parameters from", init)

    def mutate_parameters(self):
        """Randomly mutate some Gabor functions to explore new solutions"""
        if np.random.random() < self.mutation_prob:
            with torch.no_grad():
                # Randomly select 5% of Gabors to mutate
                num_gabors = self.model.amplitude.shape[0]
                num_mutate = max(1, int(0.01 * num_gabors))
                idx = np.random.choice(num_gabors, num_mutate, replace=False)
                
                device = self.model.u.device  # Get the current device
                
                # Reset their parameters randomly, ensuring correct device
                self.model.u.data[idx] = self.model.u.data[idx] + (torch.rand(num_mutate, device=device) * 2 - 1) * self.mutation_strength
                self.model.v.data[idx] = self.model.v.data[idx] + (torch.rand(num_mutate, device=device) * 2 - 1) * self.mutation_strength
                self.model.theta.data[idx] = self.model.theta.data[idx] + (torch.rand(num_mutate, device=device)* 2 - 1) * self.mutation_strength
                self.model.rel_sigma.data[idx] = self.model.rel_sigma.data[idx] + (torch.randn(num_mutate, device=device) * 2 - 1) * self.mutation_strength
                self.model.rel_freq.data[idx] = self.model.rel_freq.data[idx] +  (torch.randn(num_mutate, device=device) * 2 - 1) * self.mutation_strength
                self.model.psi.data[idx] = self.model.psi.data[idx] + (torch.randn(num_mutate, 3, device=device) * 2 - 1) * self.mutation_strength
                self.model.gamma.data[idx] = self.model.gamma.data[idx] + (torch.randn(num_mutate, device=device) * 2 - 1) * self.mutation_strength
                self.model.amplitude.data[idx] = self.model.amplitude.data[idx] + (torch.randn(num_mutate, 3, device=device) * 2 - 1) * self.mutation_strength

    def update_temperature(self, iteration, max_iterations):
        """Update temperature for simulated annealing"""
        self.current_temp = max(
            self.min_temp,
            self.initial_temp * (1 - iteration / max_iterations)
        )

    def weighted_loss(self, output, target, weights=None):
        """Calculate weighted MSE loss with gradient preservation"""
        if weights is None:
            weights = torch.ones_like(target[0])
        
        # Ensure tensors have gradients
        if not output.requires_grad:
            output.requires_grad_(True)
            
        # Calculate MSE with weights
        diff = (output - target) ** 2
        weighted_diff = diff * weights[None, :, :]
        loss = weighted_diff.mean()
        
        return loss
    
    def constraint_loss(self, model):
        # Vectorized pairwise constraints
        with torch.no_grad():
            rel_sigma = model.rel_sigma
            rel_freq = model.rel_freq
            gamma = model.gamma

        pairwise_constraints = torch.stack([
            (rel_sigma - rel_freq / 64).unsqueeze(0),
            (rel_freq / 2 - rel_sigma).unsqueeze(0),
            (rel_sigma - rel_freq).unsqueeze(0),
            (4 * rel_sigma - gamma).unsqueeze(0)
        ], dim=2)  # Stack along the last dimension

        # Calculate the squared constraints using ReLU
        con_sqr = torch.relu(pairwise_constraints) ** 2

        # Sum across the last dimension (k)
        con_losses = torch.mean(con_sqr, dim=2)

        # Sum across the mini-batch (n)
        con_loss_per_fit = torch.mean(con_losses, dim=1)
        con_loss = con_loss_per_fit.mean() / 100  # Use PyTorch's mean
        return con_loss
    def single_optimize(self,model_index,iterations):
        # Convert target image to tensor and normalize
        target_image_tensor = self.target.clone().detach().to(self.target.device)  # No unsqueeze
        
        #Removed normalization
        # Extract the specific model parameters to optimize
        specific_model_params = {
            'u': self.model.u[model_index].detach().clone().requires_grad_(),
            'v': self.model.v[model_index].detach().clone().requires_grad_(),
            'theta': self.model.theta[model_index].detach().clone().requires_grad_(),
            'rel_sigma': self.model.rel_sigma[model_index].detach().clone().requires_grad_(),
            'rel_freq': self.model.rel_freq[model_index].detach().clone().requires_grad_(),
            'psi': self.model.psi[model_index].detach().clone().requires_grad_(),
            'gamma': self.model.gamma[model_index].detach().clone().requires_grad_(),
            'amplitude': self.model.amplitude[model_index].detach().clone().requires_grad_()
        }    
        # Create a list of parameters to optimize
        params_to_optimize = [specific_model_params[param] for param in specific_model_params]
        # Initialize optimizer for specific parameters
        optimizer = optim.AdamW(
            params_to_optimize,
            lr=0.03,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )

        for iteration in range(iterations):
            # Zero gradients
            optimizer.zero_grad()

            # Temporarily set the model parameters to the optimized values
            with torch.no_grad():
                self.model.u[model_index] = specific_model_params['u']
                self.model.v[model_index] = specific_model_params['v']
                self.model.theta[model_index] = specific_model_params['theta']
                self.model.rel_sigma[model_index] = specific_model_params['rel_sigma']
                self.model.rel_freq[model_index] = specific_model_params['rel_freq']
                self.model.psi[model_index] = specific_model_params['psi']
                self.model.gamma[model_index] = specific_model_params['gamma']
                self.model.amplitude[model_index] = specific_model_params['amplitude']

            # Forward pass for the specific model
            output = self.model(self.grid_x, self.grid_y)
            weighted_diff = (output - target_image_tensor) ** 2 * self.weights
            loss = weighted_diff.mean()

             # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Update the model parameters with the optimized values
        with torch.no_grad():
            self.model.u[model_index] = specific_model_params['u']
            self.model.v[model_index] = specific_model_params['v']
            self.model.theta[model_index] = specific_model_params['theta']
            self.model.rel_sigma[model_index] = specific_model_params['rel_sigma']
            self.model.rel_freq[model_index] = specific_model_params['rel_freq']
            self.model.psi[model_index] = specific_model_params['psi']
            self.model.gamma[model_index] = specific_model_params['gamma']
            self.model.amplitude[model_index] = specific_model_params['amplitude']

        print(f"Optimization for model {model_index} completed. Loss: {loss.item():.6f}")
        return loss.item()

    def train_step(self, iteration, max_iterations):
        # Update temperature
        self.update_temperature(iteration, max_iterations)
        self.mutate_parameters()
        
        # Switch to local phase at the specified split point
        if iteration == int(max_iterations * self.phase_split):
            self.switch_to_local_phase()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(
            self.grid_x, 
            self.grid_y, 
            temperature=self.current_temp,
            dropout_active=(iteration < max_iterations * 0.8)
        )
        
        # Calculate loss
        weights = self.get_phase_specific_weights()
        loss = self.weighted_loss(output, self.target, weights) + self.constraint_loss(self.model)
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step(loss)
        
        return loss.item()

    def get_current_image(self, use_best=True):
        """Get current image"""
        with torch.no_grad():
            # Print key parameter stats           
            if use_best and self.best_state is not None:
                # Use the best model state for final output
                current_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.model.load_state_dict(self.best_state)
                output = self.model(self.grid_x, self.grid_y)
                self.model.load_state_dict(current_state)
            else:
                output = self.model(self.grid_x, self.grid_y)
                
            # Denormalize the output
            output = output * 0.5 + 0.5
            return output.clamp(0, 1).cpu().numpy()
        
    def resize_image(self,image,target_size):
        if isinstance(target_size, int):
            # If single number, maintain aspect ratio
            w, h = image.size
            aspect_ratio = w / h
            if w > h:
                new_w = target_size
                new_h = int(target_size / aspect_ratio)
            else:
                new_h = target_size
                new_w = int(target_size * aspect_ratio)
            target_size = (new_w, new_h)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        return image

    def get_phase_specific_weights(self):
        """Calculate phase-specific importance weights"""
        H, W = self.target.shape[-2:]
        
        # Create base weights
        weights = torch.ones((H, W), device=self.target.device)
        
        # Add gaussian-weighted center focus
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.target.device),
            torch.linspace(-1, 1, W, device=self.target.device)
        )
        r = torch.sqrt(x*x + y*y)
        gaussian_weight = torch.exp(-r * 2)
        
        if self.optimization_phase == 'global':
            # Global phase: focus more on center
            weights = weights * (0.5 + 0.5 * gaussian_weight)
        else:
            # Local phase: more uniform weights with slight center bias
            weights = weights * (0.8 + 0.2 * gaussian_weight)
        
        # Normalize weights
        weights = weights / weights.mean()
        weights = weights*self.weights
        return weights

    def switch_to_local_phase(self):
        """Switch optimization from global to local phase"""
        self.optimization_phase = 'local'
        self.optimizer = self.local_optimizer
        self.scheduler = self.local_scheduler
        
        # Reduce dropout for fine-tuning
        self.model.dropout.p = 0.0001
        
        # Update mutation settings
        self.mutation_prob = 0.0001
        
        print("Switching to local optimization phase...")

    def save_model(self, path):
        """Save the model state with parameter info"""
        state_dict = self.model.state_dict()
        torch.save(state_dict, path)
        print(f"Saved model to {path}")

    def save_weights(self, path):
        with torch.no_grad():
            params = {
                    'u': self.model.u.cpu().tolist(),
                    'v': self.model.v.cpu().tolist(),
                    'theta': self.model.theta.cpu().tolist(),
                    'rel_sigma': self.model.rel_sigma.cpu().tolist(),
                    'rel_freq': self.model.rel_freq.cpu().tolist(),
                    'psi0': self.model.psi[:,0].cpu().tolist(),
                    'psi1': self.model.psi[:,1].cpu().tolist(),
                    'psi2': self.model.psi[:,2].cpu().tolist(),
                    'gamma': self.model.gamma.cpu().tolist(),
                    'amplitude0': self.model.amplitude[:,0].cpu().tolist(),
                    'amplitude1': self.model.amplitude[:,1].cpu().tolist(),
                    'amplitude2': self.model.amplitude[:,2].cpu().tolist()
                }
            par = np.array([params['u'], params['v'], params['theta'], params['rel_sigma'], params['gamma'], params['rel_freq'], params['psi0'], params['psi1'], params['psi2'], params['amplitude0'], params['amplitude1'], params['amplitude2']])
            flat = par.transpose()
            np.savetxt(path, flat,fmt='%f', delimiter=',')

    def load_model(self, path):
        """Load the model state with parameter verification"""
        state_dict = torch.load(path)  
        self.model.load_state_dict(state_dict)
        print(f"Loaded model from {path}")
    
    def save_image(self, path):
        """Save the current image to a file"""
        image = self.get_current_image()
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        Image.fromarray(image).save(path)

def main():
    """Run Gabor image fitting on an input image."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fit image with Gabor functions')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--weight', type=str, help='Path to weight image (grayscale)', default=None)
    parser.add_argument('--num-gabors', type=int, default=256,
                       help='Number of Gabor functions to fit')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of training iterations')
    parser.add_argument('--single-iterations', type=int, default=100,
                       help='Number of training iterations')
    parser.add_argument('--init', type=str, default=None,
                       help='load initial parameters from file')
    parser.add_argument('--init-size', type=int, default=128,
                       help='size of image used for initialization')
    # Add size arguments
    parser.add_argument('--size', type=int, default=None,
                       help='Target size (maintains aspect ratio)')
    parser.add_argument('--width', type=int, default=None,
                       help='Target width')
    parser.add_argument('--height', type=int, default=None,
                       help='Target height')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory for output files')
    parser.add_argument('--phase-split', type=float, default=0.25,
                       help='Fraction of iterations to spend in global phase (0-1)')
    parser.add_argument('--global-lr', type=float, default=0.03,
                       help='Learning rate for global phase')
    parser.add_argument('--local-lr', type=float, default=0.01,
                       help='Learning rate for local phase')
    parser.add_argument('--mutation-strength', type=float, default=0.01,
                       help='Mutation strength')
    parser.add_argument('--sigma-max', type=float, default=2,
                       help='Maximum sigma')
    parser.add_argument('--gamma-max', type=float, default=2,
                       help='Maximum gamma')
    parser.add_argument('--rel-freq-max', type=float, default=2,
                       help='Maximum rel freq')
    parser.add_argument('--rel-freq-min', type=float, default=0.01,
                       help='Minimum rel freq')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine target size
    target_size = None
    if args.size is not None:
        target_size = args.size
    elif args.width is not None and args.height is not None:
        target_size = (args.width, args.height)
    init_gabors = args.num_gabors

    GABOR_MAX['rel_sigma'] = args.sigma_max
    GABOR_MAX['gamma'] = args.gamma_max
    GABOR_MAX['rel_freq'] = args.rel_freq_max
    GABOR_MIN['rel_freq'] = args.rel_freq_min
    # Initialize fitter with target size and learning rates
    fitter = ImageFitter(
        args.image, 
        args.weight, 
        init_gabors, 
        target_size, 
        args.device, 
        args.init,
        global_lr=args.global_lr,
        local_lr=args.local_lr,
        init_size=args.init_size,
        mutation_strength=args.mutation_strength
    )
    # Set the phase split from arguments
    fitter.phase_split = args.phase_split
    #save initial image
    if args.init:
        fitter.save_image(os.path.join(args.output_dir, 'initial_result.png'))

    # Training loop
    print(f"Training on {args.device}...")
    quat = int(args.num_gabors/4)
    accum_filters = 0
    with tqdm(total=args.iterations) as pbar:
        progress = 0
        print("Full Optimization")
        for m in range(args.num_gabors):
            fitter.single_optimize(m,10,target_image)
        for i in range(args.iterations):
            loss = fitter.train_step(i, args.iterations)    
            if i % 10 == 0:
                temp = fitter.current_temp
                pbar.set_postfix(loss=f"{loss:.6f}", temp=f"{temp:.3f}")
                pbar.update(10)
            if i % 50 == 0 or i == args.iterations - 1:
                    fitter.save_image(os.path.join(args.output_dir, f'result_{progress:04d}.png'))            
                    progress+=1
        
    # Save final result
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        fitter.save_image(os.path.join(args.output_dir, 'final_result.png'))
        fitter.save_model(os.path.join(args.output_dir, 'saved_model.pth'))
        fitter.save_weights(os.path.join(args.output_dir, 'saved_weights.txt'))
if __name__ == '__main__':
    main()


