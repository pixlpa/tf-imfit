import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import models
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
    'amplitude': 0.01
}

GABOR_MAX ={
    'u': 1,
    'v': 1,
    'theta': 2,
    'rel_sigma': 0.7,
    'rel_freq': 5,
    'gamma': 0.15,
    'psi': 3,
    'amplitude': 0.5
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
        u = self.u
        v = self.v
        theta = self.theta*2*np.pi
        sigma = self.rel_sigma
        gamma = self.gamma
        cr = torch.cos(theta[:,None,None])
        sr = torch.sin(theta[:,None,None])
        
        # Compute rotated coordinates
        x_rot = (grid_x[None,:,:] - u[:,None,None]) * cr + \
                (grid_y[None,:,:] - v[:,None,None]) * sr
        y_rot = -(grid_x[None,:,:] - u[:,None,None]) * sr + \
                (grid_y[None,:,:] - v[:,None,None]) * cr
        
        gaussian = torch.exp(
            -(x_rot**2)/(2*(sigma[:,None,None]**2)) - (y_rot**2)/(2*(gamma[:,None,None]**2))
        )
        
        # Safe sinusoid computation with frequency scaling
        freq = np.float32(2*np.pi) / torch.exp(self.rel_freq)
        phase = self.psi*2*np.pi
        sinusoid = torch.cos(freq[:,None,None,None] * x_rot[:, None, :, :] + 
                           phase[:, :, None, None])
        
        # Combine components safely
        gabors = self.amplitude[:, :, None, None] * gaussian[:, None, :, :] * sinusoid
        result = torch.sum(gabors, dim=0)  # This should be [num_gabors, height, width]

        result = torch.clamp(result, -1, 1)  # Clamp to normalized range       
        return result
    def enforce_parameter_ranges(self):
        """Enforce valid parameter ranges"""
        with torch.no_grad():
            self.u.clamp_(-1, 1)
            self.v.clamp_(-1, 1)
            self.theta.clamp_(-2, 2)
            self.rel_sigma.clamp_(1e-5,5)
            self.rel_freq.clamp_(1e-5,20)
            self.psi.clamp_(-1, 1)
            self.gamma.clamp_(1e-5,5)
            self.amplitude.clamp_(0,1)
    def mutate(self,strength):
        """Randomly mutate some Gabor functions to explore new solutions"""
        with torch.no_grad():
            # Randomly select 5% of Gabors to mutate
            num_gabors = self.amplitude.shape[0]
            num_mutate = max(1, int(0.01 * num_gabors))
            idx = np.random.choice(num_gabors, num_mutate, replace=False)
            
            device = self.u.device  # Get the current device
            
            # Reset their parameters randomly, ensuring correct device
            self.u.data[idx] = self.u.data[idx] + (torch.rand(num_mutate, device=device) * 2 - 1) * strength
            self.v.data[idx] = self.v.data[idx] + (torch.rand(num_mutate, device=device) * 2 - 1) * strength
            self.theta.data[idx] = self.theta.data[idx] + (torch.rand(num_mutate, device=device)* 2 - 1) * strength
            self.rel_sigma.data[idx] = self.rel_sigma.data[idx] + (torch.randn(num_mutate, device=device) * 2 - 1) * strength
            self.rel_freq.data[idx] = self.rel_freq.data[idx] +  (torch.randn(num_mutate, device=device) * 2 - 1) * strength
            self.psi.data[idx] = self.psi.data[idx] + (torch.randn(num_mutate, 3, device=device) * 2 - 1) * strength
            self.gamma.data[idx] = self.gamma.data[idx] + (torch.randn(num_mutate, device=device) * 2 - 1) * strength
            self.amplitude.data[idx] = self.amplitude.data[idx] + (torch.randn(num_mutate, 3, device=device) * 2 - 1) * strength


class ImageFitter:
    def __init__(self, image_path, weight_path=None, num_gabors=256, target_size=None, 
                 device='cuda' if torch.cuda.is_available() else 'cpu', init=None,
                 global_lr=0.03, local_lr=0.01, init_size=128, mutation_strength=0.01):  # Add learning rate parameters
        image = Image.open(image_path).convert('RGB')
        
        # Resize image if target_size is specified
        if target_size is not None:
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
        
        # Enhanced preprocessing pipeline
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.global_lr = global_lr
        self.local_lr = local_lr

        self.target = transform(image).to(device)
        h, w = self.target.shape[-2:]
        if target_size is not None:
            w,h = target_size
        
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
        
        # Initialize model with improved training setup
        self.model = GaborLayer(num_gabors,init_size).to(device)
        # Initialize model parameters if provided
        if init:
            self.init_parameters(init)
        # Initialize optimizers with provided learning rates
        self.global_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=global_lr,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        self.optimizer = self.global_optimizer  # Start with global optimizer
        
        # Initialize schedulers for both phases
        self.global_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)
        
        self.scheduler = self.global_scheduler  # Start with global scheduler
        
        # Use a combination of MSE and L1 loss
        self.mse_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()
        
        # Initialize best loss tracking
        self.best_loss = float('inf')
        self.best_state = None
        
        # Add temperature scheduling
        self.initial_temp = 1
        self.min_temp = 0.001
        self.current_temp = self.initial_temp
        
        # Add mutation probability
        self.mutation_prob = 0.001
        self.mutation_strength = mutation_strength
        # Add phase tracking
        self.optimization_phase = 'global'  # 'global' or 'local'
        self.phase_split = 0.5  # default value, will be updated from args
    
    def add_gabor(self, count, iterations, target_image, lr = 0.003):
        """Add a new Gabor filter to the model."""
        target_image_tensor = target_image.clone().detach().to(self.target.device)
        new_gabor = GaborLayer(num_gabors=1, base_scale=self.model.base_scale).to(self.model.u.device)
        check_loss = 0.0
        loss_diff = 0.0
        optimizer = optim.AdamW(
            new_gabor.parameters(),
            lr=lr,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        for iteration in range(iterations):
            # Zero gradients
            optimizer.zero_grad() 
            # Forward pass of both the existing model and additional gabor
            output = self.model(self.grid_x, self.grid_y) + new_gabor(self.grid_x, self.grid_y)
            weighted_diff = (output - target_image_tensor) ** 2 * self.weights

            laplace = self.lap_loss(output, self.target)
            unweighted = self.unweighted_loss(output, self.target)
            loss = weighted_diff.mean()*0.5 + laplace*0.25 + unweighted*0.25

             # Backward pass and optimize
            loss.backward()
            optimizer.step()
            loss_diff = abs(loss.item()-check_loss)
            check_loss = loss.item()
            # print(f"Local optimization of model {count}: loss {check_loss:.6f} diff {loss_diff:.6f}")
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
        self.optimizer = self.global_optimizer
        return check_loss

    def init_parameters(self, init):
        """Initialize parameters from a saved model"""
        if init:
            self.load_model(init)
            print("Initialized parameters from", init)

    def init_optimizer(self,global_lr):# Initialize optimizers with provided learning rates
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=global_lr,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        # Initialize schedulers for both phases
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)

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
    
    def unweighted_loss(self, output, target):
        """Calculate weighted MSE loss with gradient preservation"""
        # Ensure tensors have gradients
        if not output.requires_grad:
            output.requires_grad_(True)
            
        # Calculate MSE with weights
        mse  = self.mse_criterion(output,target)
        l1 = self.l1_criterion(output,target)
        loss = mse * 0.5 + 0.5 * l1
        
        return loss
    
    def lap_loss(self, output, target):
       #  print("Output shape:", output.shape)
       # print("Target shape:", target.shape)
        outp = output.unsqueeze(0)
        targ = target.unsqueeze(0)
        laplacian = nn.Conv2d(
            in_channels=3,  # 3 channels for RGB images
            out_channels=3,  # Output will also have 3 channels
            kernel_size=3,
            padding=1,
            bias=False
        )
        
        # Initialize the weights for the Laplacian filter
        laplacian.weight.data =torch.tensor([[
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]],
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]],
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]]
        ]]).float().to(target.device)  # Shape will be [1, 3, 3, 3]

        # Assign the weights and set requires_grad to False
        laplacian.weight.requires_grad = False

        output_lap = laplacian(outp)
        target_lap = laplacian(targ)
        return nn.functional.mse_loss(output_lap, target_lap)
        
    def constraint_loss(self, model):
        # Vectorized pairwise constraints
        with torch.no_grad():
            rel_sigma = model.rel_sigma
            rel_freq = model.rel_freq
            gamma = model.gamma

        pairwise_constraints = torch.stack([
            (rel_sigma - rel_freq / 32).unsqueeze(0),
            (rel_freq / 2 - rel_sigma).unsqueeze(0),
            (rel_sigma - rel_freq).unsqueeze(0),
            (8 * rel_sigma - gamma).unsqueeze(0)
        ], dim=2)  # Stack along the last dimension

        # Calculate the squared constraints using ReLU
        con_sqr = torch.relu(pairwise_constraints) ** 2

        # Sum across the last dimension (k)
        con_losses = torch.mean(con_sqr, dim=2)

        # Sum across the mini-batch (n)
        con_loss_per_fit = torch.mean(con_losses, dim=1)
        con_loss = con_loss_per_fit.mean() / 50  # Use PyTorch's mean
        return con_loss

    def train_step(self, iteration, max_iterations):
        # Update temperature
        self.update_temperature(iteration, max_iterations)
        self.mutate_parameters()
        
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
        weighted = self.weighted_loss(output, self.target, self.weights)*0.5
        unweighted = self.unweighted_loss(output, self.target)*0.4
        laplace = self.lap_loss(output,self.target) * 0.1

        loss =  weighted + unweighted + laplace # + self.constraint_loss(self.model)
        # loss = self.unweighted_loss(output, self.target) + self.perceptual_loss(output,self.target) + self.constraint_loss(self.model)
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step(loss)
        
        return loss.item()

    def get_current_image(self, use_best=True):
        """Get current image with parameter state logging"""
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

    def save_parameters_to_text(self, filepath):
        """Save model parameters in a human-readable format"""
        with torch.no_grad():
            params = {
                'u': self.model.u.cpu().tolist(),
                'v': self.model.v.cpu().tolist(),
                'theta': self.model.theta.cpu().tolist(),
                'rel_sigma': self.model.rel_sigma.cpu().tolist(),
                'rel_freq': self.model.rel_freq.cpu().tolist(),
                'psi': self.model.psi.cpu().tolist(),
                'gamma': self.model.gamma.cpu().tolist(),
                'amplitude': self.model.amplitude.cpu().tolist()
            }
            
            with open(filepath, 'w') as f:
                f.write("Gabor Function Parameters:\n\n")
                for i in range(len(params['u'])):
                    f.write(f"Gabor {i}:\n")
                    f.write(f"  Position (u, v): ({params['u'][i]:.4f}, {params['v'][i]:.4f})\n")
                    f.write(f"  Orientation (θ): {params['theta'][i]:.4f}\n")
                    f.write(f"  Size (σ): {params['rel_sigma'][i]:.4f}\n")
                    f.write(f"  Wavelength (λ): {1.0 / params['rel_freq'][i]:.4f}\n")
                    f.write(f"  Phase (ψ): {[f'{a:.4f}' for a in params['psi'][i]]}\n")
                    f.write(f"  Aspect ratio (γ): {params['gamma'][i]:.4f}\n")
                    f.write(f"  Amplitude (RGB): {[f'{a:.4f}' for a in params['amplitude'][i]]}\n")
                    f.write("\n")

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
    parser.add_argument('--single-iterations', type=int, default=20,
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
    parser.add_argument('--single-lr', type=float, default=0.1,
                       help='Learning rate for single gabor optimization')
    parser.add_argument('--mutation-strength', type=float, default=0.01,
                       help='Mutation strength')
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
    if args.init:
        init_gabors = args.num_gabors
    else:
        init_gabors = 1

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
    if args.init:
        fitter.save_image(os.path.join(args.output_dir, 'initial_result.png'))
    # Training loop
    print(f"Training on {args.device}...")
    quat = int(args.num_gabors/4)
    accum_filters = 0
    with tqdm(total=args.iterations) as pbar:
        progress = 0
        if not args.init:
            print("Building up model")
            for i in range(args.num_gabors):
                if (i > 0):
                    loss = fitter.add_gabor(i, args.single_iterations,fitter.target)  # Add a new Gabor filter
                    print(f"Added Gabor {i} with loss {loss:.6f}")
                    fitter.save_image(os.path.join(args.output_dir, f'result_{progress:04d}.png'))
                progress+=1
                if i % 32 == 0 and i > 0:
                    print("Intermediate Optimization")
                    fitter.init_optimizer(args.global_lr)
                    for i in range(int(args.iterations/4)):
                        loss = fitter.train_step(i, args.iterations)    
                        if i % 10 == 0:
                            temp = fitter.current_temp
                            pbar.set_postfix(loss=f"{loss:.6f}", temp=f"{temp:.3f}")
                            pbar.update(10)
                        if i % 20 == 0 or i == args.iterations - 1:
                                fitter.save_image(os.path.join(args.output_dir, f'result_{progress:04d}.png'))            
                                progress+=1
            fitter.init_optimizer(args.global_lr)
            for i in range(args.iterations):
                loss = fitter.train_step(i, args.iterations)    
                if i % 10 == 0:
                    temp = fitter.current_temp
                    pbar.set_postfix(loss=f"{loss:.6f}", temp=f"{temp:.3f}")
                    pbar.update(10)
                if i % 100 == 0 or i == args.iterations - 1:
                        fitter.save_image(os.path.join(args.output_dir, f'result_{progress:04d}.png'))            
                        progress+=1
        else:
            print("Optimizing full model")
            for i in range(args.iterations):
                loss = fitter.train_step(i, args.iterations)
                if i % 10 == 0:
                    temp = fitter.current_temp
                    pbar.set_postfix(loss=f"{loss:.6f}", temp=f"{temp:.3f}")
                    pbar.update(10)
            
                # Save intermediate results
                if i % 50 == 0 or i == args.iterations - 1:
                    fitter.save_image(os.path.join(args.output_dir, f'result_{progress:04d}.png'))
                    progress+=1
        
    # Save final result
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        fitter.save_image(os.path.join(args.output_dir, 'final_result.png'))
        fitter.save_model(os.path.join(args.output_dir, 'saved_model.pth'))
        fitter.save_weights(os.path.join(args.output_dir, 'saved_weights.txt'))
        fitter.save_parameters_to_text(os.path.join(args.output_dir, 'parameters.txt'))

if __name__ == '__main__':
    main()


