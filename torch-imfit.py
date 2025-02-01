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

class GaborLayer(nn.Module):
    def __init__(self, num_gabors=256):
        super().__init__()
        
        # Initialize parameters with even better defaults
        self.u = nn.Parameter(torch.rand(num_gabors) * 2 - 1)  # [-1, 1]
        self.v = nn.Parameter(torch.rand(num_gabors) * 2 - 1)  # [-1, 1]
        self.theta = nn.Parameter(torch.rand(num_gabors) * np.pi)  # [0, π]
        # Wider range of initial sizes
        self.sigma = nn.Parameter(torch.randn(num_gabors) * 0.7 - 1.0)  # log-space, more varied sizes
        self.lambda_ = nn.Parameter(torch.randn(num_gabors) * 0.9)  # log-space, more varied frequencies
        self.psi = nn.Parameter(torch.rand(num_gabors) * 2 * np.pi)  # [0, 2π]
        self.gamma = nn.Parameter(torch.randn(num_gabors) * 0.2)  # slightly elliptical Gabors
        # Initialize amplitudes with color correlation
        self.amplitude = nn.Parameter(torch.randn(num_gabors, 3) * 0.05)
        self.dropout = nn.Dropout(p=0.01)  # Add dropout

    def forward(self, grid_x, grid_y, temperature=1.0, dropout_active=True):
        # Convert parameters to proper ranges
        u = self.u * 2 - 1  # [-1, 1]
        v = self.v * 2 - 1  # [-1, 1] 
        theta = self.theta * 2 * np.pi  # [0, 2π]
        sigma = torch.exp(self.sigma)  # (0, ∞)
        lambda_ = torch.exp(self.lambda_)  # (0, ∞)
        psi = self.psi * 2 * np.pi  # [0, 2π]
        gamma = torch.exp(self.gamma)  # (0, ∞)
        amplitude = self.amplitude  # [0, 1] for each channel

        # Add random noise during training
        if self.training:
            u = u + torch.randn_like(u) * 0.02 * temperature
            v = v + torch.randn_like(v) * 0.02 * temperature
            theta = theta + torch.randn_like(theta) * 0.02 * temperature
            
        # Compute rotated coordinates for each Gabor
        x_rot = (grid_x[None,:,:] - u[:,None,None]) * torch.cos(theta[:,None,None]) + \
                (grid_y[None,:,:] - v[:,None,None]) * torch.sin(theta[:,None,None])
        y_rot = -(grid_x[None,:,:] - u[:,None,None]) * torch.sin(theta[:,None,None]) + \
                (grid_y[None,:,:] - v[:,None,None]) * torch.cos(theta[:,None,None])

        # Compute Gabor functions
        gaussian = torch.exp(-(x_rot**2 + (gamma[:,None,None] * y_rot)**2) / (2 * sigma[:,None,None]**2))
        sinusoid = torch.cos(2 * np.pi * x_rot / lambda_[:,None,None] + psi[:,None,None])
        
        # Modified: compute Gabor functions for each color channel
        gabors = amplitude[:,:,None,None] * gaussian[:, None, :, :] * sinusoid[:, None, :, :]
        
        # Apply dropout during training if requested
        if dropout_active and self.training:
            gabors = self.dropout(gabors)
        
        return torch.sum(gabors, dim=0)  # Returns shape [3, H, W]

class ImageFitter:
    def __init__(self, image_path, weight_path=None, num_gabors=256, device='cuda' if torch.cuda.is_available() else 'cpu'):
        image = Image.open(image_path).convert('RGB')
        
        # Enhanced preprocessing pipeline
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.target = transform(image).to(device)
        h, w = self.target.shape[-2:]
        
        # Load and process weight image if provided
        if weight_path:
            weight_img = Image.open(weight_path).convert('L')  # Convert to grayscale
            weight_img = weight_img.resize((w, h), Image.Resampling.LANCZOS)
            self.weights = transforms.ToTensor()(weight_img).to(device)
            # Normalize weights to average to 1
            self.weights = self.weights / self.weights.mean()
        else:
            self.weights = torch.ones_like(self.target[0]).to(device)  # Just channel 0 for shape

        # Create coordinate grid
        y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
        self.grid_x = x.to(device)
        self.grid_y = y.to(device)
        
        # Initialize model with improved training setup
        self.model = GaborLayer(num_gabors).to(device)
        
        # Use AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.03,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # More aggressive learning rate scheduling
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=50,
            verbose=True,
            min_lr=1e-5
        )
        
        # Use a combination of MSE and L1 loss
        self.mse_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()
        
        # Initialize best loss tracking
        self.best_loss = float('inf')
        self.best_state = None
        
        # Add temperature scheduling
        self.initial_temp = 0.1
        self.min_temp = 0.0001
        self.current_temp = self.initial_temp
        
        # Add mutation probability
        self.mutation_prob = 0.001

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
                self.model.sigma.data[idx] = (torch.randn(num_mutate, device=device) * 0.7 - 1.0)
                self.model.lambda_.data[idx] = (torch.randn(num_mutate, device=device) * 0.7)
                self.model.psi.data[idx] = (torch.rand(num_mutate, device=device) * 2 * np.pi)
                self.model.gamma.data[idx] = (torch.randn(num_mutate, device=device) * 0.2)
                self.model.amplitude.data[idx] = (torch.randn(num_mutate, 3, device=device) * 0.05)

    def update_temperature(self, iteration, max_iterations):
        """Update temperature for simulated annealing"""
        self.current_temp = max(
            self.min_temp,
            self.initial_temp * (1 - iteration / max_iterations)
        )

    def weighted_loss(self, output, target, weights):
        # Apply weights to each channel
        mse_per_pixel = (output - target).pow(2)
        l1_per_pixel = torch.abs(output - target)
        
        # Sum across channels, then apply weights
        mse_loss = (mse_per_pixel.mean(dim=0) * weights).mean()
        l1_loss = (l1_per_pixel.mean(dim=0) * weights).mean()
        
        return mse_loss + 0.1 * l1_loss

    def train_step(self, iteration, max_iterations):
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
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory for output files')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize fitter
    fitter = ImageFitter(args.image, args.weight, args.num_gabors, args.device)

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
    final_result = fitter.get_current_image()
    final_result = np.transpose(final_result, (1, 2, 0))
    final_result = np.clip(final_result * 255, 0, 255).astype(np.uint8)
    final_img = Image.fromarray(final_result)
    final_img.save(os.path.join(args.output_dir, 'final_result.png'))

if __name__ == '__main__':
    main()


