"""
Adapter module to integrate optimal augmentation generation into the distillation pipeline.
This module replaces predefined augmentations (rotations) with learned optimal augmentations.
"""

import sys
import os
import numpy as np
import torch

# Add path to optimal augmentation code
optimal_aug_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'optimal-data-augmentation-ssl')
if optimal_aug_path not in sys.path:
    sys.path.insert(0, optimal_aug_path)

from src.augmentation_generator import BarlowTwinsAugmentationGenerator
from src.kernels import get_kernel


class OptimalAugmentationGenerator:
    """
    Wrapper class to generate optimal augmentations for distilled dataset.
    Replaces predefined rotation augmentations with learned optimal augmentations.
    """
    
    def __init__(self, kernel_type='rbf', kernel_params=None, lambda_ridge=1.0, mu_p=1.0):
        """
        Initialize the optimal augmentation generator.
        
        Args:
            kernel_type: Type of kernel to use ('rbf', 'linear', 'polynomial')
            kernel_params: Dictionary of kernel-specific parameters
            lambda_ridge: Ridge regularization parameter
            mu_p: Pre-image solver parameter
        """
        if kernel_params is None:
            if kernel_type == 'rbf':
                kernel_params = {'gamma': 1e-5}
            elif kernel_type == 'polynomial':
                kernel_params = {'degree': 2, 'coef0': 1.0}
            else:
                kernel_params = {}
        
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params
        self.lambda_ridge = lambda_ridge
        self.mu_p = mu_p
        self.generator = None
        self.num_augmentations = 3  # Default to match 3 rotation augmentations
        self.image_shape = None
        
    def fit(self, images_np, representations_np):
        """
        Fit the optimal augmentation generator.
        
        Args:
            images_np: numpy array of shape (num_images, channels, height, width)
            representations_np: numpy array of shape (num_images, feature_dim)
        
        Returns:
            self
        """
        print(f"Fitting optimal augmentation generator with {self.kernel_type} kernel...")
        
        # Store image shape for later reconstruction
        self.image_shape = images_np.shape[1:]  # (channels, height, width)
        
        # Flatten images to matrix format (features x samples)
        # Original shape: (n, c, h, w) -> flatten to (c*h*w, n)
        num_images = images_np.shape[0]
        X_train = images_np.reshape(num_images, -1).T  # Shape: (c*h*w, n)
        
        # Transpose representations to (feature_dim, n)
        F_target = representations_np.T
        
        print(f"  Image matrix shape: {X_train.shape}")
        print(f"  Target representation shape: {F_target.shape}")
        
        # Create kernel
        kernel = get_kernel(self.kernel_type, **self.kernel_params)
        
        # Create and fit generator
        self.generator = BarlowTwinsAugmentationGenerator(
            kernel=kernel,
            lambda_ridge=self.lambda_ridge,
            mu_p=self.mu_p,
            check_conditions=True
        )
        
        self.generator.fit(X_train, F_target)
        
        print("Optimal augmentation generator fitted successfully!")
        return self
    
    def transform(self, num_augmentations=None, indices=None):
        """
        Generate optimal augmentations for the fitted data.
        
        Args:
            num_augmentations: Number of augmented versions to generate per image
            indices: Indices of samples to augment (if None, augment all)
        
        Returns:
            List of augmented image arrays, each of shape (num_images, channels, height, width)
        """
        if self.generator is None:
            raise RuntimeError("Generator not fitted. Call fit() first.")
        
        if num_augmentations is None:
            num_augmentations = self.num_augmentations
        
        print(f"Generating {num_augmentations} optimal augmentations...")
        
        augmented_images_list = []
        
        for aug_idx in range(num_augmentations):
            print(f"  Generating augmentation {aug_idx + 1}/{num_augmentations}...")
            
            # Generate augmented images
            X_augmented = self.generator.transform(indices=indices)
            
            # Reshape back to image format
            # X_augmented shape: (c*h*w, n) -> (n, c, h, w)
            num_samples = X_augmented.shape[1]
            images_aug = X_augmented.T.reshape(num_samples, *self.image_shape)
            augmented_images_list.append(images_aug)
        
        print("Optimal augmentations generated successfully!")
        return augmented_images_list
    
    def transform_torch(self, images_torch, num_augmentations=None):
        """
        Apply optimal augmentations to torch tensor images.
        This method is used during training to augment dynamically updated images.
        
        Args:
            images_torch: Torch tensor of shape (num_images, channels, height, width)
            num_augmentations: Number of augmentations to generate (default: 3)
        
        Returns:
            List of augmented image tensors
        """
        if self.generator is None:
            raise RuntimeError("Generator not fitted. Call fit() first.")
        
        if num_augmentations is None:
            num_augmentations = self.num_augmentations
        
        # Convert to numpy
        images_np = images_torch.detach().cpu().numpy()
        device = images_torch.device
        
        # Flatten images to matrix format
        num_images = images_np.shape[0]
        X = images_np.reshape(num_images, -1).T
        
        augmented_tensors = []
        for aug_idx in range(num_augmentations):
            # Generate augmented images using fitted generator
            X_augmented = self.generator.transform(indices=None)
            
            # Reshape back to image format
            images_aug = X_augmented.T.reshape(num_images, *images_np.shape[1:])
            
            # Convert to torch tensor
            images_aug_torch = torch.from_numpy(images_aug).float().to(device)
            augmented_tensors.append(images_aug_torch)
        
        return augmented_tensors
    
    def generate_augmented_representations(self, teacher_model, images_np, device='cuda'):
        """
        Generate augmented images and their corresponding representations.
        
        Args:
            teacher_model: Pre-trained teacher model to extract representations
            images_np: Original images as numpy array (num_images, channels, height, width)
            device: Device to run computations on
        
        Returns:
            augmented_images_list: List of augmented images (numpy arrays)
            augmented_reprs_list: List of augmented representations (numpy arrays)
        """
        # First, get representations of original images
        images_torch = torch.from_numpy(images_np).float().to(device)
        with torch.no_grad():
            reprs_orig = teacher_model(images_torch).cpu().numpy()
        
        # Fit the generator
        self.fit(images_np, reprs_orig)
        
        # Generate augmented images
        augmented_images_list = self.transform(num_augmentations=self.num_augmentations)
        
        # Get representations for augmented images
        augmented_reprs_list = []
        teacher_model.eval()
        
        for aug_images in augmented_images_list:
            aug_images_torch = torch.from_numpy(aug_images).float().to(device)
            with torch.no_grad():
                aug_reprs = teacher_model(aug_images_torch).cpu().numpy()
            augmented_reprs_list.append(aug_reprs)
        
        return augmented_images_list, augmented_reprs_list


# Global generator instance to reuse during training
_global_generator = None


def get_optimal_augmentations_for_distillation(
    sample_images, 
    teacher_model, 
    device='cuda',
    kernel_type='rbf',
    kernel_params=None,
    lambda_ridge=1.0,
    mu_p=1.0,
    num_augmentations=3
):
    """
    Convenience function to generate optimal augmentations for distillation.
    
    Args:
        sample_images: Torch tensor of sampled images (num_images, channels, height, width)
        teacher_model: Pre-trained teacher model
        device: Device for computation
        kernel_type: Kernel type for optimal augmentation
        kernel_params: Kernel parameters
        lambda_ridge: Ridge regularization
        mu_p: Pre-image solver parameter
        num_augmentations: Number of augmentations to generate
    
    Returns:
        augmented_reprs_list: List of augmented representations as numpy arrays
    """
    global _global_generator
    
    # Convert to numpy
    images_np = sample_images.cpu().numpy()
    
    # Create generator
    generator = OptimalAugmentationGenerator(
        kernel_type=kernel_type,
        kernel_params=kernel_params,
        lambda_ridge=lambda_ridge,
        mu_p=mu_p
    )
    generator.num_augmentations = num_augmentations
    
    # Generate augmentations and their representations
    _, augmented_reprs_list = generator.generate_augmented_representations(
        teacher_model, images_np, device
    )
    
    # Store generator globally for use during training
    _global_generator = generator
    
    return augmented_reprs_list


def apply_optimal_augmentations(images_torch, num_augmentations=3):
    """
    Apply fitted optimal augmentations to images during training.
    Uses the globally fitted generator from initialization phase.
    
    Args:
        images_torch: Torch tensor of images to augment
        num_augmentations: Number of augmentations to generate
    
    Returns:
        List of augmented image tensors
    """
    global _global_generator
    
    if _global_generator is None:
        raise RuntimeError("Optimal augmentation generator not initialized. Call get_optimal_augmentations_for_distillation first.")
    
    return _global_generator.transform_torch(images_torch, num_augmentations)

