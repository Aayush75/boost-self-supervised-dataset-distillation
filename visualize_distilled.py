"""
Visualize Distilled Dataset Images

This script loads the distilled dataset and creates a grid visualization of the 
synthesized images, similar to Figure 2 in the paper.
"""

import os
import yaml
import argparse
import torch
import numpy as np

# Force matplotlib to use non-interactive backend (no GUI required)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from main_distill import DistilledData


def denormalize_cifar10(img):
    """Denormalize CIFAR-10 images."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    return img * std + mean


def denormalize_cifar100(img):
    """Denormalize CIFAR-100 images."""
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
    return img * std + mean


def denormalize_stanford_dogs(img):
    """Denormalize Stanford Dogs images."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img * std + mean


def get_denormalize_fn(dataset_name):
    """Get the appropriate denormalization function for a dataset."""
    if dataset_name.upper() == "CIFAR10":
        return denormalize_cifar10
    elif dataset_name.upper() == "CIFAR100":
        return denormalize_cifar100
    elif dataset_name.upper() == "STANFORD_DOGS":
        return denormalize_stanford_dogs
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def visualize_distilled_images(config_path, output_path=None, grid_size=None):
    """
    Load distilled images and create a grid visualization.
    
    Args:
        config_path (str): Path to the configuration YAML file
        output_path (str): Path to save the visualization (optional)
        grid_size (tuple): Grid dimensions (rows, cols). If None, auto-calculated
    """
    print("="*80)
    print("VISUALIZING DISTILLED DATASET")
    print("="*80)
    
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_name = config['data']['name']
    asset_dir = config['saving']['distilled_assets_dir']
    num_images = config['distillation']['num_distilled_images_m']
    
    print(f"Dataset: {dataset_name}")
    print(f"Number of distilled images: {num_images}")
    print(f"Asset directory: {asset_dir}")
    
    # Check if distilled assets exist
    distilled_data_path = os.path.join(asset_dir, 'distilled_data.pth')
    if not os.path.exists(distilled_data_path):
        print(f"Error: Distilled data not found at {distilled_data_path}")
        print("Please run main_distill.py first to generate the distilled dataset.")
        return
    
    # Load distilled data
    print("Loading distilled data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the saved state dict
    state_dict = torch.load(distilled_data_path, map_location=device)
    
    # Extract shapes from the saved parameters to initialize the module correctly
    B_x_shape = state_dict['B_x'].shape
    B_y_shape = state_dict['B_y'].shape
    C_x_shape = state_dict['C_x'].shape
    C_y_shape = state_dict['C_y'].shape
    num_aug = len([k for k in state_dict.keys() if k.startswith('C_aug_y')])
    
    # Create properly-sized initialization parameters from the saved state
    init_params = {
        'B_x': state_dict['B_x'].cpu().numpy(),
        'B_y': state_dict['B_y'].cpu().numpy(),
        'C_x': state_dict['C_x'].cpu().numpy(),
        'C_y': state_dict['C_y'].cpu().numpy(),
        'C_aug_y': [state_dict[f'C_aug_y.{i}'].cpu().numpy() for i in range(num_aug)]
    }
    
    distilled_data = DistilledData(init_params, config).to(device)
    distilled_data.eval()
    
    # Reconstruct images
    print("Reconstructing distilled images...")
    with torch.no_grad():
        images = distilled_data.reconstruct_images()
    
    # Move to CPU and denormalize
    images = images.cpu()
    denormalize_fn = get_denormalize_fn(dataset_name)
    images = denormalize_fn(images)
    
    # Clamp to [0, 1] range
    images = torch.clamp(images, 0, 1)
    
    # Calculate grid size if not provided
    if grid_size is None:
        # Try to make a square or near-square grid
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        grid_size = (rows, cols)
    
    rows, cols = grid_size
    
    print(f"Creating {rows}×{cols} grid visualization...")
    
    # Create figure
    fig_width = cols * 1.0
    fig_height = rows * 1.0
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Remove spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    
    # Flatten axes for easy iteration
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each image
    for idx in range(rows * cols):
        ax = axes[idx]
        ax.axis('off')
        
        if idx < num_images:
            # Get image and convert to numpy (H, W, C)
            img = images[idx].permute(1, 2, 0).numpy()
            ax.imshow(img)
        else:
            # Fill remaining cells with white
            ax.imshow(np.ones_like(images[0].permute(1, 2, 0).numpy()))
    
    # Add title
    title = f"{dataset_name} Distilled Images (N={num_images})"
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"Visualization saved to: {output_path}")
    else:
        # Auto-generate filename
        output_path = os.path.join(asset_dir, f'distilled_images_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"Visualization saved to: {output_path}")
    
    plt.close()
    
    print("="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize distilled dataset images in a grid layout'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration YAML file (e.g., configs/cifar100.yaml)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save the visualization (default: auto-generated in asset directory)'
    )
    parser.add_argument(
        '--grid',
        type=str,
        default=None,
        help='Grid size as "rows,cols" (e.g., "10,10"). If not specified, auto-calculated.'
    )
    
    args = parser.parse_args()
    
    # Parse grid size if provided
    grid_size = None
    if args.grid:
        try:
            rows, cols = map(int, args.grid.split(','))
            grid_size = (rows, cols)
        except:
            print(f"Warning: Invalid grid format '{args.grid}'. Using auto-calculated grid.")
    
    visualize_distilled_images(args.config, args.output, grid_size)


if __name__ == "__main__":
    main()
