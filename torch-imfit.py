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
        
        # Learnable parameters for each Gabor function
        self.u = nn.Parameter(torch.rand(num_gabors))  # x position
        self.v = nn.Parameter(torch.rand(num_gabors))  # y position
        self.theta = nn.Parameter(torch.rand(num_gabors))  # rotation
        self.sigma = nn.Parameter(torch.rand(num_gabors))  # width
        self.lambda_ = nn.Parameter(torch.rand(num_gabors))  # wavelength
        self.psi = nn.Parameter(torch.rand(num_gabors))  # phase
        self.gamma = nn.Parameter(torch.rand(num_gabors))  # aspect ratio
        self.amplitude = nn.Parameter(torch.rand(num_gabors, 3))  # amplitude for RGB channels

    def forward(self, grid_x, grid_y):
        # Convert parameters to proper ranges
        u = self.u * 2 - 1  # [-1, 1]
        v = self.v * 2 - 1  # [-1, 1] 
        theta = self.theta * 2 * np.pi  # [0, 2π]
        sigma = torch.exp(self.sigma)  # (0, ∞)
        lambda_ = torch.exp(self.lambda_)  # (0, ∞)
        psi = self.psi * 2 * np.pi  # [0, 2π]
        gamma = torch.exp(self.gamma)  # (0, ∞)
        amplitude = self.amplitude  # [0, 1] for each channel

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
        
        # Sum all Gabor functions for each channel
        return torch.sum(gabors, dim=0)  # Returns shape [3, H, W]

class ImageFitter:
    def __init__(self, image_path, num_gabors=256, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # Modified: Load color image and convert to RGB
        image = Image.open(image_path).convert('RGB')
        self.target = transforms.ToTensor()(image).to(device)
        h, w = self.target.shape[-2:]
        
        # Create coordinate grid
        y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
        self.grid_x = x.to(device)
        self.grid_y = y.to(device)
        
        # Initialize model and optimizer
        self.model = GaborLayer(num_gabors).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def train_step(self):
        self.optimizer.zero_grad()
        output = self.model(self.grid_x, self.grid_y)
        loss = self.criterion(output, self.target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_current_image(self):
        with torch.no_grad():
            return self.model(self.grid_x, self.grid_y).cpu().numpy()

def main():
    """Run Gabor image fitting on an input image."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fit image with Gabor functions')
    parser.add_argument('image', type=str, help='Path to input image')
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
    fitter = ImageFitter(args.image, args.num_gabors, args.device)

    # Training loop
    print(f"Training on {args.device}...")
    with tqdm(total=args.iterations) as pbar:
        for i in range(args.iterations):
            loss = fitter.train_step()
            
            if i % 10 == 0:
                pbar.set_postfix(loss=f"{loss:.6f}")
                pbar.update(10)

            # Save intermediate result every 100 iterations
            if i % 100 == 0:
                result = fitter.get_current_image()
                # Convert to correct format and range for PIL
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


