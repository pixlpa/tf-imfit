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

class GaborLayer(nn.Module):
    def __init__(self, num_gabors=256):
        super().__init__()
        
        # All parameters normalized/scale-independent:
        self.u = nn.Parameter(torch.rand(num_gabors) * 2 - 1)  # [-1, 1] position
        self.v = nn.Parameter(torch.rand(num_gabors) * 2 - 1)  # [-1, 1] position
        self.theta = nn.Parameter(torch.rand(num_gabors) * np.pi)  # [0, π] rotation
        
        # Convert size parameters to relative/normalized form
        self.rel_sigma = nn.Parameter(torch.randn(num_gabors) * 1.0 - 1.0)  # relative to image size
        self.rel_freq = nn.Parameter(torch.randn(num_gabors) * 0.7)  # relative frequency
        
        self.psi = nn.Parameter(torch.randn(num_gabors,3) * 2 * np.pi)  # [0, 2π] phase
        self.gamma = nn.Parameter(torch.randn(num_gabors) * 0.4)  # aspect ratio
        self.amplitude = nn.Parameter(torch.randn(num_gabors, 3) * 0.06)  # strength
        self.dropout = nn.Dropout(p=0.01)

    def forward(self, grid_x, grid_y, temperature=1.0, dropout_active=True):
        # Get image dimensions from grid
        H, W = grid_x.shape
        image_size = max(H, W)
        
        # Convert parameters to proper ranges
        u = self.u * 2 - 1  # [-1, 1]
        v = self.v * 2 - 1  # [-1, 1] 
        theta = self.theta * 2 * np.pi  # [0, 2π]
        
        # Convert relative parameters to absolute
        sigma = torch.exp(self.rel_sigma) * (image_size / 32)  # Scale relative to image size
        freq = torch.exp(self.rel_freq) * (32 / image_size)    # Inverse scale for frequency
        lambda_ = 1.0 / freq  # Convert frequency to wavelength
        
        gamma = torch.exp(self.gamma)
        
        # Add random noise during training
        if self.training:
            u = u + torch.randn_like(u) * 0.0002 * temperature
            v = v + torch.randn_like(v) * 0.0002 * temperature
            theta = theta + torch.randn_like(theta) * 0.0002 * temperature
            
        # Compute rotated coordinates for each Gabor
        x_rot = (grid_x[None,:,:] - u[:,None,None]) * torch.cos(theta[:,None,None]) + \
                (grid_y[None,:,:] - v[:,None,None]) * torch.sin(theta[:,None,None])
        y_rot = -(grid_x[None,:,:] - u[:,None,None]) * torch.sin(theta[:,None,None]) + \
                (grid_y[None,:,:] - v[:,None,None]) * torch.cos(theta[:,None,None])

        # Compute Gabor functions
        gaussian = torch.exp(-(x_rot**2 + (gamma[:,None,None] * y_rot)**2) / (2 * sigma[:,None,None]**2))
        
        # Modified: handle psi for each color channel separately
        sinusoid = torch.cos(2 * np.pi * x_rot[:, None, :, :] / lambda_[:, None, None, None] + self.psi[:, :, None, None])
        
        # Compute Gabor functions for each color channel
        gabors = self.amplitude[:,:,None,None] * gaussian[:, None, :, :] * sinusoid
        
        # Apply dropout during training if requested
        if dropout_active and self.training:
            gabors = self.dropout(gabors)
        
        return torch.sum(gabors, dim=0)  # Returns shape [3, H, W]

class ImageFitter:
    def __init__(self, image_path, weight_path=None, num_gabors=256, target_size=None, 
                 device='cuda' if torch.cuda.is_available() else 'cpu', init=None,
                 global_lr=0.03, local_lr=0.01):  # Add learning rate parameters
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
        
        self.target = transform(image).to(device)
        h, w = self.target.shape[-2:]
        if target_size is not None:
            w,h = target_size
        
        # Enhanced weight calculation if no weight_path provided
        if weight_path:
            weight_img = Image.open(weight_path).convert('L')  # Convert to grayscale
            weight_img = weight_img.resize((w, h), Image.Resampling.LANCZOS)
            self.weights = transforms.ToTensor()(weight_img).to(device)
            # Normalize weights to average to 1
            self.weights = self.weights / self.weights.mean()
        else:
            # Calculate edge-based weights using Sobel filters
            target_np = self.target.cpu().numpy()
            gray = np.mean(target_np, axis=0)  # Convert to grayscale
            
            # Apply Sobel filters
            dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            dy = dx.T
            grad_x = np.abs(scipy.signal.convolve2d(gray, dx, mode='same', boundary='symm'))
            grad_y = np.abs(scipy.signal.convolve2d(gray, dy, mode='same', boundary='symm'))
            
            # Combine gradients and normalize
            edge_weights = np.sqrt(grad_x**2 + grad_y**2)
            edge_weights = edge_weights / edge_weights.mean()
            
            # Add bias to ensure non-edge regions still get some weight
            edge_weights = 0.5 + 0.5 * edge_weights
            
            self.weights = torch.from_numpy(edge_weights).float().to(device)

        # Create coordinate grid
        y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
        self.grid_x = x.to(device)
        self.grid_y = y.to(device)
        
        # Initialize model with improved training setup
        self.model = GaborLayer(num_gabors).to(device)
        # Initialize model parameters if provided
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
        
        # Use a combination of MSE and L1 loss
        self.mse_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()
        
        # Initialize best loss tracking
        self.best_loss = float('inf')
        self.best_state = None
        
        # Add temperature scheduling
        self.initial_temp = 0.08
        self.min_temp = 0.00000001
        self.current_temp = self.initial_temp
        
        # Add mutation probability
        self.mutation_prob = 0.001
        
        # Add phase tracking
        self.optimization_phase = 'global'  # 'global' or 'local'
        self.phase_split = 0.5  # default value, will be updated from args

    def init_parameters(self, init):
        if init is not None:
            load_model(self, init)

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
                self.model.u.data[idx] = (torch.rand(num_mutate, device=device) * 2 - 1)
                self.model.v.data[idx] = (torch.rand(num_mutate, device=device) * 2 - 1)
                self.model.theta.data[idx] = (torch.rand(num_mutate, device=device) * np.pi)
                self.model.rel_sigma.data[idx] = (torch.randn(num_mutate, device=device) * 0.7 - 1.0)
                self.model.rel_freq.data[idx] = (torch.randn(num_mutate, device=device) * 0.7)
                self.model.psi.data[idx] = (torch.randn(num_mutate, 3, device=device) * 2 * np.pi)
                self.model.gamma.data[idx] = (torch.randn(num_mutate, device=device) * 0.2)
                self.model.amplitude.data[idx] = (torch.randn(num_mutate, 3, device=device) * 0.05)

    def update_temperature(self, iteration, max_iterations):
        """Update temperature for simulated annealing"""
        self.current_temp = max(
            self.min_temp,
            self.initial_temp * (1 - iteration / max_iterations)
        )

    def weighted_loss(self, output, target, weights):
        # Calculate multiple loss components
        
        # 1. Per-pixel losses
        mse_per_pixel = (output - target).pow(2)
        l1_per_pixel = torch.abs(output - target)
        
        # 2. Structural similarity
        # Calculate local means and variances for structural comparison
        kernel_size = 11
        padding = kernel_size // 2
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(output.device) / (kernel_size * kernel_size)
        
        # Calculate means
        output_mean = torch.nn.functional.conv2d(
            output.unsqueeze(0), kernel.expand(3, -1, -1, -1), 
            padding=padding, groups=3
        ).squeeze(0)
        target_mean = torch.nn.functional.conv2d(
            target.unsqueeze(0), kernel.expand(3, -1, -1, -1), 
            padding=padding, groups=3
        ).squeeze(0)
        
        # Calculate variances and covariance
        output_var = torch.nn.functional.conv2d(
            (output.unsqueeze(0) - output_mean.unsqueeze(0)).pow(2),
            kernel.expand(3, -1, -1, -1),
            padding=padding, groups=3
        ).squeeze(0)
        target_var = torch.nn.functional.conv2d(
            (target.unsqueeze(0) - target_mean.unsqueeze(0)).pow(2),
            kernel.expand(3, -1, -1, -1),
            padding=padding, groups=3
        ).squeeze(0)
        
        # Structural similarity term
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = ((2 * output_mean * target_mean + c1) * 
                (2 * torch.sqrt(output_var * target_var) + c2)) / \
               ((output_mean.pow(2) + target_mean.pow(2) + c1) * 
                (output_var + target_var + c2))
        
        # 3. Edge detection loss
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              device=output.device).float().unsqueeze(0).unsqueeze(0)
        sobel_y = sobel_x.transpose(2, 3)
        
        # Calculate edges
        output_edges_x = torch.nn.functional.conv2d(output.mean(dim=0, keepdim=True).unsqueeze(0), 
                                                  sobel_x, padding=1)
        output_edges_y = torch.nn.functional.conv2d(output.mean(dim=0, keepdim=True).unsqueeze(0), 
                                                  sobel_y, padding=1)
        target_edges_x = torch.nn.functional.conv2d(target.mean(dim=0, keepdim=True).unsqueeze(0), 
                                                  sobel_x, padding=1)
        target_edges_y = torch.nn.functional.conv2d(target.mean(dim=0, keepdim=True).unsqueeze(0), 
                                                  sobel_y, padding=1)
        
        edge_loss = torch.mean(torch.abs(
            torch.sqrt(output_edges_x.pow(2) + output_edges_y.pow(2)) -
            torch.sqrt(target_edges_x.pow(2) + target_edges_y.pow(2))
        ))
        
        # Combine losses with weights
        pixel_loss = (
            0.5 * (mse_per_pixel.mean(dim=0) * weights).mean() +  # MSE loss
            0.5 * (l1_per_pixel.mean(dim=0) * weights).mean()     # L1 loss
        )
        structural_loss = (1 - ssim.mean(dim=0) * weights).mean()
        
        # Phase-specific loss weighting
        if self.optimization_phase == 'global':
            # In global phase, focus more on structural similarity
            total_loss = (
                0.7 * pixel_loss +
                0.2 * structural_loss +
                0.1 * edge_loss
            )
        else:
            # In local phase, increase weight of edge and pixel losses
            total_loss = (
                0.8 * pixel_loss +
                0.1 * structural_loss +
                0.1 * edge_loss
            )
        
        return total_loss

    def train_step(self, iteration, max_iterations):
        # Switch to local phase at the specified split point
        if iteration == int(max_iterations * self.phase_split):
            self.switch_to_local_phase()
        
        self.optimizer.zero_grad()
        
        # Update temperature
        self.update_temperature(iteration, max_iterations)
        
        # Forward pass with current temperature
        output = self.model(
            self.grid_x, 
            self.grid_y, 
            temperature=self.current_temp,
            dropout_active=(iteration < max_iterations * 0.8)  # Disable dropout near end
        )
        
        # Calculate loss
        loss = self.weighted_loss(output, self.target, self.weights)
        
        # Add random mutation
        self.mutate_parameters()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        self.scheduler.step(loss)
        
        # Save best state (only when temperature is low)
        if self.current_temp < 0.5 and loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        return loss.item()

    def get_current_image(self, use_best=True):
        with torch.no_grad():
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

    def get_phase_specific_weights(self):
        """Get weights specific to current optimization phase"""
        if self.optimization_phase == 'global':
            # For global phase, use smoother weights
            weights = self.weights.clone()
            # Apply Gaussian blur to weights
            kernel_size = 9
            sigma = 2.0
            weights = gaussian_blur(
                weights.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
                kernel_size=kernel_size,
                sigma=sigma
            ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions
            return weights
        else:
            # For local phase, use sharp weights and enhance high-frequency regions
            weights = self.weights.clone()
            # Enhance edge weights
            weights = weights ** 1.5
            return weights

    def switch_to_local_phase(self):
        """Switch optimization from global to local phase"""
        self.optimization_phase = 'local'
        self.optimizer = self.local_optimizer
        self.scheduler = self.local_scheduler
        
        # Reduce dropout for fine-tuning
        self.model.dropout.p = 0.0001
        
        # Update mutation settings
        self.mutation_prob = 0.00005
        
        print("Switching to local optimization phase...")

    def save_model(self, path):
        """Save the model state with parameter info"""
        state_dict = self.model.state_dict()
        print("\nSaving model parameters:")
        for name, param in state_dict.items():
            print(f"{name}: shape {param.shape}, range [{param.min():.3f}, {param.max():.3f}]")
        torch.save(state_dict, path)
        print(f"Saved model to {path}")

    def load_model(self, path):
        """Load the model state with parameter verification"""
        state_dict = torch.load(path)
        print("\nLoading model parameters:")
        for name, param in state_dict.items():
            print(f"{name}: shape {param.shape}, range [{param.min():.3f}, {param.max():.3f}]")
        
        # Print current model parameters before loading
        print("\nCurrent model parameters before loading:")
        for name, param in self.model.state_dict().items():
            print(f"{name}: shape {param.shape}, range [{param.min():.3f}, {param.max():.3f}]")
            
        self.model.load_state_dict(state_dict)
        print(f"Loaded model from {path}")

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
    parser.add_argument('--init', type=str, default=None,
                       help='load initial parameters from file')
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
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine target size
    target_size = None
    if args.size is not None:
        target_size = args.size
    elif args.width is not None and args.height is not None:
        target_size = (args.width, args.height)

    # Initialize fitter with target size and learning rates
    fitter = ImageFitter(
        args.image, 
        args.weight, 
        args.num_gabors, 
        target_size, 
        args.device, 
        args.init,
        global_lr=args.global_lr,
        local_lr=args.local_lr
    )
    # Set the phase split from arguments
    fitter.phase_split = args.phase_split

    # Training loop
    print(f"Training on {args.device}...")
    with tqdm(total=args.iterations) as pbar:
        for i in range(args.iterations):
            loss = fitter.train_step(i, args.iterations)
            
            if i % 10 == 0:
                temp = fitter.current_temp
                pbar.set_postfix(loss=f"{loss:.6f}", temp=f"{temp:.3f}")
                pbar.update(10)
            
            # Save intermediate results
            if i % 100 == 0:
                result = fitter.get_current_image()
                result = np.transpose(result, (1, 2, 0))
                result = np.clip(result * 255, 0, 255).astype(np.uint8)
                img = Image.fromarray(result)
                img.save(os.path.join(args.output_dir, f'result_{i:04d}.png'))
    # Save final result
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        final_result = fitter.get_current_image()
        final_result = np.transpose(final_result, (1, 2, 0))
        final_result = np.clip(final_result * 255, 0, 255).astype(np.uint8)
        final_img = Image.fromarray(final_result)
        final_img.save(os.path.join(args.output_dir, 'final_result.png'))
        fitter.save_model(os.path.join(args.output_dir, 'saved_model.pth'))
        fitter.save_parameters_to_text(os.path.join(args.output_dir, 'parameters.txt'))

if __name__ == '__main__':
    main()


