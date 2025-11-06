"""
Simple utility to generate optimal augmentations for sampled distilled images.
Fits generator on the 100 sampled images (fast - 100x100 kernel) during initialization.
"""

import os
import sys
import numpy as np
import torch

# Add path to optimal augmentation code
optimal_aug_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'optimal-data-augmentation-ssl')
if optimal_aug_path not in sys.path:
    sys.path.insert(0, optimal_aug_path)

from src.augmentation_generator import BarlowTwinsAugmentationGenerator
from src.kernels import get_kernel
from src.utils import images_to_matrix, matrix_to_images


def generate_optimal_augmentations_for_init(images_torch, teacher_model, kernel_config, device='cuda', num_augmentations=3):
    """
    Generate optimal augmentations for initialization.
    Fits generator on the provided images (typically 100 sampled images).
    
    Args:
        images_torch: Torch tensor of images (num_images, channels, height, width)
        teacher_model: Pre-trained teacher model to extract target representations
        kernel_config: Dict with kernel_type, kernel_params, lambda_ridge, mu_p
        device: Device for computation
        num_augmentations: Number of augmentations to generate
        
    Returns:
        List of augmented representations as numpy arrays (for C_aug_y initialization)
    """
    print(f"\nGenerating optimal augmentations for {images_torch.shape[0]} sampled images...")
    print(f"Kernel: {kernel_config['kernel_type']}, Params: {kernel_config.get('kernel_params', {})}")
    
    # Convert to numpy
    images_np = images_torch.cpu().numpy()
    
    # Extract target representations
    print("Extracting target representations...")
    teacher_model.eval()
    with torch.no_grad():
        target_reprs = teacher_model(images_torch.to(device)).cpu().numpy()
    
    # Convert to matrix format for generator
    X_train = images_to_matrix(images_np)  # (features, num_images)
    F_target = target_reprs.T  # (feature_dim, num_images)
    
    print(f"Image matrix shape: {X_train.shape}")
    print(f"Target representation shape: {F_target.shape}")
    
    # Create kernel
    kernel_type = kernel_config.get('kernel_type', 'rbf')
    kernel_params = kernel_config.get('kernel_params', {'gamma': 1e-5})
    kernel = get_kernel(kernel_type, **kernel_params)
    
    # Fit generator (fast for 100 images - 100x100 kernel)
    print(f"Fitting augmentation generator (this takes ~10-30 seconds for 100 images)...")
    generator = BarlowTwinsAugmentationGenerator(
        kernel=kernel,
        lambda_ridge=kernel_config.get('lambda_ridge', 1.0),
        mu_p=kernel_config.get('mu_p', 1.0),
        check_conditions=True
    )
    
    generator.fit(X_train, F_target)
    print("Generator fitted successfully!")
    
    # Get info
    aug_info = generator.get_augmentation_distribution()
    print(f"Augmentation info - Condition number: {aug_info['condition_number']:.2e}")
    
    # Generate augmentations and get their representations
    print(f"Generating {num_augmentations} augmentations...")
    augmented_reprs_list = []
    
    for aug_idx in range(num_augmentations):
        # Generate augmented images
        X_augmented = generator.transform(indices=None)
        
        # Convert back to image format
        images_aug = matrix_to_images(X_augmented, image_shape=images_np.shape[1:])
        
        # Clip to valid range and convert to torch
        images_aug = np.clip(images_aug, 0, 1)
        images_aug_torch = torch.from_numpy(images_aug).float().to(device)
        
        # Get representations
        with torch.no_grad():
            aug_reprs = teacher_model(images_aug_torch).cpu().numpy()
        
        augmented_reprs_list.append(aug_reprs)
        print(f"  Generated augmentation {aug_idx + 1}/{num_augmentations}")
    
    print("Optimal augmentation generation complete!")
    return augmented_reprs_list

