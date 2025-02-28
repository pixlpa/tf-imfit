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

# set the min and max initialization values
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

#the Model definition
class GaborLayer(nn.Module):
    def __init__(self, num_gabors=256):
        super().__init__()
        
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

    def forward(self, grid_x, grid_y):
        H, W = grid_x.shape
        image_size = max(H, W)

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
            self.rel_sigma.clamp_(1e-4,5)
            self.rel_freq.clamp_(1e-7,2)
            self.psi.clamp_(-1, 1)
            self.gamma.clamp_(1e-4,1.5)
            self.amplitude.clamp_(0,1)

# The ImageFitter class manages the work of training the model
class ImageFitter:
    def __init__(self, image_path, weight_path=None, num_gabors=256, target_size=None, 
                 device='cuda' if torch.cuda.is_available() else 'cpu', init=None,
                 global_lr=0.03, local_lr=0.01, init_size=128, mutation_strength=0.01, gamma = 0.85,
                 sobel = 0., gradient = 0., l1 = 0.):  # Add learning rate parameters
        #load the image
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
        
        # preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # store these arguments as parameters to use elsewhere
        self.global_lr = global_lr
        self.gamma = gamma
        self.sobel = sobel
        self.gradient = gradient
        self.l1 = l1

        #convert image to tensor and send to CUDA
        self.target = transform(image).to(device)
        h, w = self.target.shape[-2:]
        if target_size is not None:
            w,h = target_size
        #prepare the weights tensor
        if weight_path:
            weight_img = Image.open(weight_path).convert('L')  # Convert to grayscale
            weight_img = weight_img.resize((w, h), Image.Resampling.LANCZOS)
            self.weights = transforms.ToTensor()(weight_img).to(device)
            # Normalize weights to average to 1
            self.weights = self.weights / self.weights.mean()
        else:
            self.weights = torch.ones((h, w), device=device).unsqueeze(0) # adding a dimension for resizing later

        self.og_target = self.target #store the largest versions
        self.og_weights = self.weights
        # Create coordinate grid
        y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
        self.grid_x = x.to(device)
        self.grid_y = y.to(device)
        
        # Initialize model
        self.model = GaborLayer(num_gabors).to(device)
        # Initialize model parameters if file is provided
        if init:
            self.init_parameters(init)
        # Initialize optimizers with provided learning rates
        self.global_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=global_lr,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        self.optimizer = self.global_optimizer
        
        # Initialize scheduler
        self.global_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=gamma,
            patience=20
        )
        
        self.scheduler = self.global_scheduler  # Start with global scheduler
        
        # Use a combination of MSE and L1 loss
        self.mse_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()
        
        # Initialize best loss tracking
        self.best_loss = float('inf')
        self.best_state = None
        
        # Add temperature scheduling
        # this might not do anything anymore
        self.initial_temp = 0.1
        self.min_temp = 0.001
        self.current_temp = self.initial_temp
        
        # Add mutation probability
        self.mutation_prob = 0.1
        self.mutation_strength = mutation_strength
    
    # experimental single model optimization prototype. Not used anymore
    def single_optimize(self,model_index,iterations):
        # Convert target image to tensor and normalize
        target_image_tensor = self.target.clone().detach().to(self.target.device)  # No unsqueeze
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
            loss = self.loss_function(output,self.target)

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

    def init_parameters(self, init):
        """Initialize parameters from a saved model"""
        if init:
            # self.load_model(init)
            self.load_weights(init)
            print("Initialized parameters from", init)

    def init_optimizer(self,global_lr):
        # Initialize optimizers with provided learning rates
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=global_lr,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            self.optimizer,
                            mode='min',
                            factor=self.gamma,
                            patience=20,
                            verbose=True
                        )

    def mutate_parameters(self):
        """Randomly mutate some Gabor functions to explore new solutions"""
        if np.random.random() < self.mutation_prob * self.current_temp:
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
        """Calculate L1 loss with gradient preservation"""
        # Ensure tensors have gradients
        if not output.requires_grad:
            output.requires_grad_(True)
            
        # Calculate MSE with weights
       # mse  = self.mse_criterion(output,target)
        return self.l1_criterion(output,target)
    
    def sobel_filter(self, image):
        """Perform Sobel filtering to emphasize edges"""
        # Ensure image is in the right format (B, C, H, W)
        image = image.unsqueeze(0).float()
    # Define base Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32)
        
        sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32)
        
        # Reshape kernels for 3-channel input
        # Shape: (out_channels, in_channels/groups, kernel_height, kernel_width)
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(image.device)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(image.device)
        
        # Create convolutional layers
        grad_x = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            padding=1,
            groups=3,  # Important: use groups=3 for separate filtering of each channel
            bias=False
        ).to(image.device)
        
        grad_y = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            padding=1,
            groups=3,  # Important: use groups=3 for separate filtering of each channel
            bias=False
        ).to(image.device)

        sobel_x.requires_grad = False
        sobel_y.requires_grad = False
        
        grad_x.weight.data = sobel_x
        grad_y.weight.data = sobel_y

        mag_x = grad_x(image)
        mag_y = grad_y(image)
        
        # Compute gradient magnitude
        gradient_magnitude = torch.sqrt(mag_x**2 + mag_y**2 + 1e-6)
        return gradient_magnitude
    
    def sobel_loss(self, output, target, weights):
        outs = self.sobel_filter(output)
        targ = self.sobel_filter(target)
        return nn.functional.mse_loss(outs,targ)
    
    def lap_loss(self, output, target):
        """Laplacian filtered loss, similar to sobel"""
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
    def get_gradients(self, image):
        h_gradient = image[..., :, 1:] - image[..., :, :-1]
        v_gradient = image[..., 1:, :] - image[..., :-1, :]
        return h_gradient, v_gradient
    
    def gradient_loss(self, generated_image, target_image):
        # Compute gradients for both images        
        gen_h, gen_v = self.get_gradients(generated_image)
        target_h, target_v = self.get_gradients(target_image)
        
        # Compute L1 loss between gradients
        h_loss = torch.mean(torch.abs(gen_h - target_h))
        v_loss = torch.mean(torch.abs(gen_v - target_v))
        return h_loss + v_loss   
    
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
        con_loss = con_loss_per_fit.mean() / 100  # Use PyTorch's mean
        return con_loss
    
    def loss_function(self, output, target):
        weighted = self.weighted_loss(output, target, self.weights)
        unweighted = 0
        sobel = 0
        gradient = 0
        if self.l1 > 0:
            unweighted = self.unweighted_loss(output, target) * self.l1
        # laplace = self.lap_loss(output,target) * 0.1
        if self.gradient > 0:
            gradient = self.gradient_loss(output,target) * self.gradient
        if self.sobel > 0:
            sobel = self.sobel_loss(output,target, self.weights) * self.sobel
        loss =  weighted + unweighted + gradient + sobel + self.constraint_loss(self.model)
        return loss

    def train_step(self, iteration, max_iterations, save_best = False):
        """Main Training function"""
        # Update temperature
        self.update_temperature(iteration, max_iterations)
        self.mutate_parameters()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(
            self.grid_x, 
            self.grid_y, 
            dropout_active=False
        )
        
        # Calculate loss
        loss =  self.loss_function(output, self.target)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(loss)
        
        # Store best result if save_best is active (currently only during last phase)
        if loss.item() < self.best_loss and save_best:
            self.best_loss = loss.item()
            self.best_state = self.model.state_dict()

        return loss.item()

    def get_current_image(self, use_best=False):
        """Get current image at 512 resolution"""
        h, w = self.og_target.shape[-2:]
        h1 = h
        w1 = w
        # rescale output size to 512 base
        if h > w:
            h1 = 512
            w1 = int(512 * (w / h))
        else:
            w1 = 512
            h1 = int(512 * (h/w))

        with torch.no_grad():
            y, x = torch.meshgrid(torch.linspace(-1, 1, h1), torch.linspace(-1, 1, w1))
            grid_x = x.to(self.target.device)
            grid_y = y.to(self.target.device)
            if use_best:
                self.model.load_state_dict(self.best_state)
            output = self.model(grid_x, grid_y)
            # Denormalize the output
            output = output * 0.5 + 0.5
        return output.clamp(0, 1).cpu().numpy()

    def save_model(self, path):
        """Save the model state with parameter info"""
        state_dict = self.model.state_dict()
        torch.save(state_dict, path)
        print(f"Saved model to {path}")

    def save_weights(self, path):
        """Save as comma separated data in text file"""
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
        """Load the model state file (not used)"""
        state_dict = torch.load(path)  
        self.model.load_state_dict(state_dict)
        print(f"Loaded model from {path}")
    
    def load_weights(self,path):
        """load from a saved text file used by init"""
        weights = np.genfromtxt(path, dtype=float, delimiter=",").transpose()
        device = self.model.u.device
        with torch.no_grad():
            self.model.u.data = torch.from_numpy(weights[0]).to(device)
            self.model.v.data = torch.from_numpy(weights[1]).to(device)
            self.model.theta.data = torch.from_numpy(weights[2]).to(device)
            self.model.rel_sigma.data = torch.from_numpy(weights[3]).to(device)
            self.model.gamma.data = torch.from_numpy(weights[4]).to(device)
            self.model.rel_freq.data = torch.from_numpy(weights[5]).to(device)
            self.model.psi.data = torch.from_numpy(np.array([weights[6],weights[7],weights[8]]).transpose()).to(device)
            self.model.amplitude.data = torch.from_numpy(np.array([weights[9],weights[10],weights[11]]).transpose()).to(device)
        print(f"Loaded weights from {path}")
            
    def save_image(self, path):
        """Save the current image to a file"""
        image = self.get_current_image()
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        Image.fromarray(image).save(path)
    
    def save_final(self, path):
        """Save the current image to a file"""
        image = self.get_current_image(use_best=True) #only difference is to "use_best"
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        Image.fromarray(image).save(path)
    
    def resize_target(self,size):
        # Resize target image
        if isinstance(size, int):
            h, w = self.target.shape[-2:]
            if h > w:
                new_h, new_w = size, int(size * w / h)
            else:
                new_h, new_w = int(size * h / w), size
        else:
            new_h = size
            new_w = size
            
        self.target = nn.functional.interpolate(
            self.og_target.unsqueeze(0), 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)

        weights_4d = self.og_weights.unsqueeze(0)
        if len(weights_4d.shape) < 4:
            weights_4d.squeeze(0)
        self.weights = nn.functional.interpolate(
            weights_4d,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
        
        # Update coordinate grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, new_h),
            torch.linspace(-1, 1, new_w)
        )
        self.grid_x = x.to(self.target.device)
        self.grid_y = y.to(self.target.device)

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
    parser.add_argument('--rescales', type=int, default=2,
                       help='number of scale factors to use')
    parser.add_argument('--iter-multiple', type=float, default=1.5,
                       help='number of scale factors to use')
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
    parser.add_argument('--gradient', type=float, default=0.,
                       help='weight of gradient loss function')
    parser.add_argument('--l1', type=float, default=0.,
                       help='weight of L1 loss function')
    parser.add_argument('--sobel', type=float, default=0.,
                       help='weight of sobel loss function')
    parser.add_argument('--global-lr', type=float, default=0.03,
                       help='Learning rate for global phase')
    parser.add_argument('--local-lr', type=float, default=0.01,
                       help='Learning rate for local phase')
    parser.add_argument('--mutation-strength', type=float, default=0.01,
                       help='Mutation strength')
    parser.add_argument('--gamma', type=float, default=0.85,
                       help='learning rate gamma')
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
        mutation_strength=args.mutation_strength,
        gamma = args.gamma,
        sobel = args.sobel,
        gradient = args.gradient,
        l1 = args.l1
    )
    #save initial image
    if args.init:
        fitter.save_image(os.path.join(args.output_dir, 'initial_result.png'))

    # Training loop
    print(f"Training on {args.device}...")
    progress = 0
    print("Beginning optimizations")
    fitter.init_optimizer(args.global_lr)
    #run loop for each scale
    for a in range(args.rescales):
        factor = args.rescales - a
        scaler = args.size/(2 ** factor)
        print(f"Optimizing at size: {scaler: .1f}")
        fitter.resize_target(int(scaler))
        # re-initialize scheduler for full learning rate
        for param_group in fitter.optimizer.param_groups:
            param_group['lr'] = args.global_lr
        fitter.scheduler.base_lrs = [args.global_lr]
        fitter.scheduler.last_epoch = -1
        iter_count = int( args.iterations * factor * args.iter_multiple)
        with tqdm(total=iter_count) as pbar:
            for i in range(iter_count):
                loss = fitter.train_step(i, args.iterations)    
                if i % 5 == 0:
                    temp = fitter.scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=f"{loss:.6f}", lr=f"{temp:.6f}")
                    pbar.update(5)
                if i % 50 == 0 or i == args.iterations - 1:
                        fitter.save_image(os.path.join(args.output_dir, f'result_{progress:04d}.png'))            
                        progress+=1  
    #  for b in range(20):
    #     fitter.single_optimize(np.random.randint(0, args.num_gabors-1),args.single_iterations)
    fitter.resize_target(args.size)
    for param_group in fitter.optimizer.param_groups:
        param_group['lr'] = args.global_lr
    fitter.scheduler.base_lrs = [args.global_lr]
    fitter.scheduler.last_epoch = -1
    print("Optimizing at full size")
    with tqdm(total=args.iterations) as pbar:
        for i in range(args.iterations):
            loss = fitter.train_step(i, args.iterations, save_best = True)    
            if i % 5 == 0:
                temp = fitter.scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{loss:.6f}", lr=f"{temp:.6f}")
                pbar.update(5)
            if i % 50 == 0 or i == args.iterations - 1:
                    fitter.save_image(os.path.join(args.output_dir, f'result_{progress:04d}.png'))            
                    progress+=1
    
    # Save final result
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        fitter.save_final(os.path.join(args.output_dir, 'final_result.png'))
        fitter.save_model(os.path.join(args.output_dir, 'saved_model.pth'))
        fitter.save_weights(os.path.join(args.output_dir, 'saved_weights.txt'))
        print(f"Saved final result with loss: {fitter.best_loss:.6f}")
if __name__ == '__main__':
    main()


