"""
Generate optimal augmented images for CIFAR-100 using the optimal-data-augmentation-ssl module.
This script should be run ONCE to generate the augmented images that will be used in distillation.

Run this script first:
    python generate_optimal_augmentations.py --config configs/cifar100.yaml

This will generate and save optimal augmented images to:
    ./optimal_augmentations/cifar100/
"""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add path to optimal augmentation code
optimal_aug_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'optimal-data-augmentation-ssl')
if optimal_aug_path not in sys.path:
    sys.path.insert(0, optimal_aug_path)

from src.utils import load_cifar100, images_to_matrix, matrix_to_images
from src.kernels import get_kernel
from src.target_models import TargetModel
from src.augmentation_generator import BarlowTwinsAugmentationGenerator


def generate_optimal_augmentations(config_path='configs/cifar100.yaml'):
    """
    Generate optimal augmented images for CIFAR-100 dataset.
    
    Args:
        config_path: Path to distillation config file
    """
    print("="*80)
    print("GENERATING OPTIMAL AUGMENTATIONS FOR CIFAR-100")
    print("="*80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_name = config['data']['name']
    aug_config = config.get('augmentations', {})
    
    # Set up output directory
    output_dir = f"./optimal_augmentations/{dataset_name.lower()}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Kernel type: {aug_config.get('kernel_type', 'rbf')}")
    print(f"  Number of augmentations: {aug_config.get('num_augmentations', 3)}")
    print(f"  Output directory: {output_dir}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Step 1: Load CIFAR-100 training data
    print("\n" + "="*80)
    print("STEP 1: LOADING CIFAR-100 TRAINING DATA")
    print("="*80)
    
    images, labels = load_cifar100(
        data_dir='./data',
        train=True,
        download=True,
        normalize=True,
        num_samples=None,  # Use all 50,000 training images
    )
    print(f"Loaded {len(images)} training images")
    print(f"Image shape: {images.shape}")
    
    # Convert to matrix format (features x samples)
    X_train = images_to_matrix(images)
    print(f"Data matrix shape: {X_train.shape} (features x samples)")
    
    # Step 2: Extract target representations using teacher model
    print("\n" + "="*80)
    print("STEP 2: EXTRACTING TARGET REPRESENTATIONS")
    print("="*80)
    
    teacher_path = config['models']['teacher']['path']
    feature_dim = config['models']['teacher']['feature_dim']
    
    print(f"Loading teacher model from: {teacher_path}")
    
    # Load teacher model (assuming it's a ResNet18 with Barlow Twins)
    from models import get_teacher_model
    teacher_model = get_teacher_model(teacher_path, feature_dim).to(device)
    teacher_model.eval()
    
    print("Extracting representations from teacher model...")
    all_reprs = []
    batch_size = 512
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
            batch = torch.from_numpy(images[i:i+batch_size]).float().to(device)
            reprs = teacher_model(batch).cpu().numpy()
            all_reprs.append(reprs)
    
    F_target = np.concatenate(all_reprs, axis=0)
    print(f"Target representations shape: {F_target.shape} (samples x features)")
    
    # Transpose to (features x samples) for the generator
    F_target = F_target.T
    print(f"Transposed to: {F_target.shape} (features x samples)")
    
    # Step 3: Fit optimal augmentation generator
    print("\n" + "="*80)
    print("STEP 3: FITTING OPTIMAL AUGMENTATION GENERATOR")
    print("="*80)
    
    kernel_type = aug_config.get('kernel_type', 'rbf')
    kernel_params = aug_config.get('kernel_params', {'gamma': 1e-5})
    lambda_ridge = aug_config.get('lambda_ridge', 1.0)
    mu_p = aug_config.get('mu_p', 1.0)
    
    print(f"Creating {kernel_type} kernel with params: {kernel_params}")
    kernel = get_kernel(kernel_type, **kernel_params)
    
    print(f"Initializing augmentation generator...")
    print(f"  lambda_ridge: {lambda_ridge}")
    print(f"  mu_p: {mu_p}")
    
    generator = BarlowTwinsAugmentationGenerator(
        kernel=kernel,
        lambda_ridge=lambda_ridge,
        mu_p=mu_p,
        check_conditions=True,
    )
    
    print("\nFitting generator (this may take several minutes)...")
    print("  - Computing kernel matrix K...")
    print("  - Solving kernel ridge regression...")
    print("  - Solving Lyapunov equation...")
    
    generator.fit(X_train, F_target)
    
    print("✓ Generator fitted successfully!")
    
    # Get augmentation distribution info
    aug_info = generator.get_augmentation_distribution()
    print("\nAugmentation Distribution Info:")
    print(f"  Min eigenvalue: {aug_info['min_eigenvalue']:.6e}")
    print(f"  Max eigenvalue: {aug_info['max_eigenvalue']:.6e}")
    print(f"  Condition number: {aug_info['condition_number']:.6e}")
    
    # Step 4: Generate optimal augmented images
    print("\n" + "="*80)
    print("STEP 4: GENERATING OPTIMAL AUGMENTED IMAGES")
    print("="*80)
    
    num_augmentations = aug_config.get('num_augmentations', 3)
    print(f"Generating {num_augmentations} augmented versions for all {len(images)} images...")
    
    augmented_images_list = []
    
    for aug_idx in range(num_augmentations):
        print(f"\n  Generating augmentation {aug_idx + 1}/{num_augmentations}...")
        
        # Generate augmented data matrix
        X_augmented = generator.transform(indices=None)  # Augment all samples
        
        # Convert back to image format
        images_aug = matrix_to_images(X_augmented, image_shape=(3, 32, 32))
        
        print(f"  Augmented images shape: {images_aug.shape}")
        augmented_images_list.append(images_aug)
    
    print("\n✓ All augmentations generated successfully!")
    
    # Step 5: Save augmented images
    print("\n" + "="*80)
    print("STEP 5: SAVING AUGMENTED IMAGES")
    print("="*80)
    
    # Save original images
    original_path = os.path.join(output_dir, 'original_images.npy')
    print(f"Saving original images to: {original_path}")
    np.save(original_path, images)
    
    # Save each augmentation
    for aug_idx, aug_images in enumerate(augmented_images_list):
        aug_path = os.path.join(output_dir, f'augmented_images_{aug_idx+1}.npy')
        print(f"Saving augmentation {aug_idx+1} to: {aug_path}")
        np.save(aug_path, aug_images)
    
    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'num_samples': len(images),
        'num_augmentations': num_augmentations,
        'kernel_type': kernel_type,
        'kernel_params': kernel_params,
        'lambda_ridge': lambda_ridge,
        'mu_p': mu_p,
        'image_shape': list(images.shape[1:]),
        'aug_info': {
            'min_eigenvalue': float(aug_info['min_eigenvalue']),
            'max_eigenvalue': float(aug_info['max_eigenvalue']),
            'condition_number': float(aug_info['condition_number']),
        }
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.yaml')
    print(f"Saving metadata to: {metadata_path}")
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print("\n" + "="*80)
    print("✓ OPTIMAL AUGMENTATION GENERATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {output_dir}/original_images.npy")
    for aug_idx in range(num_augmentations):
        print(f"  - {output_dir}/augmented_images_{aug_idx+1}.npy")
    print(f"  - {output_dir}/metadata.yaml")
    
    print(f"\nTotal storage size: ~{(len(images) * (num_augmentations + 1) * 3 * 32 * 32 * 4) / 1e9:.2f} GB")
    print("\nYou can now use these pre-generated augmented images in main_distill.py")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Generate optimal augmented images for CIFAR-100')
    parser.add_argument('--config', type=str, default='configs/cifar100.yaml',
                        help='Path to distillation config file')
    args = parser.parse_args()
    
    generate_optimal_augmentations(args.config)


if __name__ == '__main__':
    main()
