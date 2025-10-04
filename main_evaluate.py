import os
import yaml
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models import resnet18
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

import argparse
from utils import get_dataset
from models import get_teacher_model, InnerCNN, ApproximationMLP
from main_distill import DistilledData

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, feature_dim=512, num_classes=None):
        super().__init__()
        self.backbone = resnet18()
        self.backbone.fc = nn.Identity()
        self.feature_dim = feature_dim
        
        if num_classes is not None:
            self.classifier = nn.Linear(feature_dim, num_classes)
        else:
            self.classifier = None
    
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        if return_features or self.classifier is None:
            return features
        return self.classifier(features)
    
    def get_features(self, x):
        return self.backbone(x)

class DistilledDatasetLoader:
    def __init__(self, config, asset_dir):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load distilled data
        self.distilled_data = self._load_distilled_data(asset_dir)
        self.approx_networks = self._load_approximation_networks(asset_dir)
        
    def _load_distilled_data(self, asset_dir):
        """Load the distilled data parameters."""
        # Create dummy init params (will be overwritten by state_dict)
        init_params = {
            'B_x': torch.zeros(self.config['parametrization']['image_bases_U'], 
                              3 * self.config['parametrization']['image_basis_size']**2),
            'B_y': torch.zeros(self.config['parametrization']['repr_bases_V'], 
                              self.config['models']['inner_cnn']['feature_dim']),
            'C_x': torch.zeros(self.config['distillation']['num_distilled_images_m'], 
                              self.config['parametrization']['image_bases_U']),
            'C_y': torch.zeros(self.config['distillation']['num_distilled_images_m'], 
                              self.config['parametrization']['repr_bases_V']),
            'C_aug_y': [torch.zeros(self.config['distillation']['num_distilled_images_m'], 
                                   self.config['parametrization']['repr_bases_V']) 
                       for _ in self.config['augmentations']['rotate']]
        }
        
        distilled_data = DistilledData(init_params, self.config).to(self.device)
        state_dict_path = os.path.join(asset_dir, 'distilled_data.pth')
        distilled_data.load_state_dict(torch.load(state_dict_path, map_location=self.device))
        distilled_data.eval()
        return distilled_data
    
    def _load_approximation_networks(self, asset_dir):
        """Load the approximation networks."""
        approx_networks = []
        for rot_angle in self.config['augmentations']['rotate']:
            net = ApproximationMLP(
                num_repr_bases_V=self.config['parametrization']['repr_bases_V'],
                hidden_dim=self.config['models']['approximation_mlp']['hidden_dim']
            ).to(self.device)
            net_path = os.path.join(asset_dir, f'approx_net_rot_{rot_angle}.pth')
            net.load_state_dict(torch.load(net_path, map_location=self.device))
            net.eval()
            approx_networks.append(net)
        return approx_networks
    
    def get_distilled_data(self):
        """Get the distilled images and their target representations."""
        with torch.no_grad():
            images = self.distilled_data.reconstruct_images()
            representations = self.distilled_data.reconstruct_representations()[0]  # Base representations
        return images, representations
    
    def get_augmented_training_data(self):
        """Get all training data including base images and augmented views with their targets."""
        with torch.no_grad():
            # Get base data
            base_images = self.distilled_data.reconstruct_images()
            base_representations = self.distilled_data.reconstruct_representations()[0]
            
            all_images = [base_images]
            all_targets = [base_representations]
            
            # Get base coefficients for approximation networks
            base_coeffs = self.distilled_data.C_y
            
            # Generate augmented data
            for i, rot_angle in enumerate(self.config['augmentations']['rotate']):
                # Apply rotation to base images
                aug_images = TF.rotate(base_images, angle=float(rot_angle))
                all_images.append(aug_images)
                
                # Use approximation network to predict augmented targets
                approx_net = self.approx_networks[i]
                approx_net.eval()
                
                # Get predicted shift and compute augmented coefficients
                pred_shift = approx_net(base_coeffs)
                aug_coeffs = base_coeffs + pred_shift
                
                # Reconstruct target representations from augmented coefficients
                aug_targets = aug_coeffs @ self.distilled_data.B_y
                all_targets.append(aug_targets)
            
            # Combine all data
            train_images = torch.cat(all_images, dim=0)
            train_targets = torch.cat(all_targets, dim=0)
            
        return train_images, train_targets

def pretrain_on_full_dataset(config, save_path=None):
    """Train a ResNet18 feature extractor on the full dataset using MSE loss against the teacher."""
    print("=" * 60)
    print(f"TRAINING RESNET18 ON FULL {config['data']['name']} DATASET")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, _ = get_dataset(config['data']['name'], data_dir='.')
    
    # Load teacher model to get target representations
    teacher_model = get_teacher_model(
        config['models']['teacher']['path'], 
        config['models']['teacher']['feature_dim']
    ).to(device)
    
    # Create ResNet18 model
    model = ResNet18FeatureExtractor(feature_dim=config['models']['teacher']['feature_dim']).to(device)
    
    # Training setup
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=float(config['evaluation']['lr']), 
        momentum=0.9, 
        weight_decay=float(config['evaluation']['weight_decay'])
    )
    criterion = nn.MSELoss()
    
    # Training loop
    epochs = int(config['evaluation']['epochs'])
    print(f"Training for {epochs} epochs...")
    
    model.train()
    for epoch in tqdm(range(epochs), desc=f"Training on full {config['data']['name']}"):
        total_loss = 0
        num_batches = 0
        
        for images, _ in train_loader:
            images = images.to(device)
            
            # Get target representations from teacher
            with torch.no_grad():
                target_repr = teacher_model(images)
            
            # Forward pass
            pred_repr = model.get_features(images)
            loss = criterion(pred_repr, target_repr)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
    
    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Full dataset model saved to {save_path}")
    
    return model

def pretrain_on_distilled_dataset(config, asset_dir, save_path=None):
    """Train a ResNet18 feature extractor on the distilled dataset with augmentations and approximation networks."""
    print("=" * 60)
    print(f"TRAINING RESNET18 ON DISTILLED {config['data']['name']} DATASET (WITH AUGMENTATIONS)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load distilled data and approximation networks
    distilled_loader = DistilledDatasetLoader(config, asset_dir)
    
    # Get base distilled data for info
    base_images, base_representations = distilled_loader.get_distilled_data()
    print(f"Base distilled dataset size: {len(base_images)} images")
    print(f"Base image shape: {base_images.shape}")
    print(f"Base representation shape: {base_representations.shape}")
    print(f"Number of approximation networks: {len(distilled_loader.approx_networks)}")
    
    # Get augmented training data following the paper's methodology
    print("Preparing augmented training data using approximation networks...")
    train_images, train_targets = distilled_loader.get_augmented_training_data()
    
    total_samples = len(train_images)
    augmentation_factor = total_samples / len(base_images)
    
    print(f"Total training samples: {total_samples} (augmentation factor: {augmentation_factor:.1f}x)")
    print(f"Final training image shape: {train_images.shape}")
    print(f"Final training target shape: {train_targets.shape}")
    
    # Create ResNet18 model
    model = ResNet18FeatureExtractor(feature_dim=512).to(device)
    
    # Training setup
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=float(config['evaluation']['lr']), 
        momentum=0.9, 
        weight_decay=float(config['evaluation']['weight_decay'])
    )
    criterion = nn.MSELoss()
    
    # Training loop
    epochs = int(config['evaluation']['epochs'])
    print(f"Training for {epochs} epochs...")
    
    model.train()
    for epoch in tqdm(range(epochs), desc=f"Training on distilled {config['data']['name']} with augmentations"):
        # Forward pass on all training data (base + augmented)
        pred_repr = model.get_features(train_images)
        loss = criterion(pred_repr, train_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Distilled dataset model saved to {save_path}")
    
    return model

def linear_evaluation(model, config, split='test'):
    """
    Perform linear evaluation following SSL protocol:
    1. Freeze the feature extractor weights
    2. Train only a linear classifier (nn.Linear) on top
    3. Evaluate on test set
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Freeze the feature extractor
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    # Get datasets
    train_dataset, test_dataset = get_dataset(config['data']['name'], data_dir='.')
    print(f"Dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # Create linear classifier (only this will be trained)
    feature_dim = 512  # ResNet18 feature dimension
    num_classes = config['data']['num_classes']
    linear_classifier = nn.Linear(feature_dim, num_classes).to(device)
    
    # Training setup for linear classifier only
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(), 
        lr=float(config['evaluation']['linear_lr']), 
        momentum=0.9, 
        weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config['evaluation']['linear_epochs']))
    criterion = nn.CrossEntropyLoss()
    
    # Training loop for linear classifier
    epochs = int(config['evaluation']['linear_epochs'])
    print(f"Training linear classifier for {epochs} epochs...")
    
    for epoch in range(epochs):
        linear_classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            # Extract features (frozen)
            with torch.no_grad():
                features = model.get_features(images)
            
            # Forward pass through linear classifier only
            outputs = linear_classifier(features)
            loss = criterion(outputs, labels)
            
            # Backward pass (only linear classifier parameters)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        train_acc = 100.0 * correct / total
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
    
    # Evaluation on test set
    print("Evaluating on test set...")
    linear_classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            
            # Extract features (frozen)
            features = model.get_features(images)
            
            # Forward pass through linear classifier
            outputs = linear_classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Linear Evaluation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

def evaluate_models(config_path):
    """Main evaluation function comparing full dataset vs distilled dataset training."""
    print("Starting Model Evaluation...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    asset_dir = config['saving']['distilled_assets_dir']
    
    # Check if distilled assets exist
    if not os.path.exists(asset_dir):
        print(f"Error: Distilled assets not found at {asset_dir}")
        print("Please run main_distill.py first to generate distilled dataset.")
        return
    
    # Create save directories for models
    eval_dir = "./evaluation_models"
    full_model_path = os.path.join(eval_dir, "resnet18_full_dataset.pth")
    distilled_model_path = os.path.join(eval_dir, "resnet18_distilled_dataset.pth")
    
    results = {}
    
    # Train and evaluate model on full dataset
    print("\n" + "="*80)
    print(f"EXPERIMENT 1: ResNet18 trained on FULL {config['data']['name']} dataset")
    print("="*80)
    
    start_time = time.time()
    full_model = pretrain_on_full_dataset(config, save_path=full_model_path)
    full_training_time = time.time() - start_time
    
    print("Evaluating full dataset model...")
    full_accuracy = linear_evaluation(full_model, config, split='test')
    results['full_dataset'] = {
        'accuracy': full_accuracy,
        'training_time': full_training_time
    }
    
    # Train and evaluate model on distilled dataset
    print("\n" + "="*80)
    print(f"EXPERIMENT 2: ResNet18 trained on DISTILLED {config['data']['name']} dataset")
    print("="*80)
    
    start_time = time.time()
    distilled_model = pretrain_on_distilled_dataset(config, asset_dir, save_path=distilled_model_path)
    distilled_training_time = time.time() - start_time
    
    print("Evaluating distilled dataset model...")
    distilled_accuracy = linear_evaluation(distilled_model, config, split='test')
    results['distilled_dataset'] = {
        'accuracy': distilled_accuracy,
        'training_time': distilled_training_time
    }
    
    # Print comparison results
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS COMPARISON FOR {config['data']['name']}")
    print("="*80)
    
    print(f"Full {config['data']['name']} Dataset Training:")
    print(f"  - Test Accuracy: {results['full_dataset']['accuracy']:.4f}")
    print(f"  - Training Time: {results['full_dataset']['training_time']:.2f} seconds")
    
    print(f"\nDistilled {config['data']['name']} Dataset Training:")
    print(f"  - Test Accuracy: {results['distilled_dataset']['accuracy']:.4f}")
    print(f"  - Training Time: {results['distilled_dataset']['training_time']:.2f} seconds")
    
    # Calculate efficiency metrics
    accuracy_ratio = results['distilled_dataset']['accuracy'] / results['full_dataset']['accuracy']
    time_ratio = results['distilled_dataset']['training_time'] / results['full_dataset']['training_time']
    
    print(f"\nEfficiency Metrics:")
    print(f"  - Accuracy Retention: {accuracy_ratio:.4f} ({accuracy_ratio*100:.2f}%)")
    print(f"  - Training Time Ratio: {time_ratio:.4f} ({time_ratio*100:.2f}%)")
    
    full_dataset_size = config['data']['full_dataset_size']
    storage_budget = config['distillation']['storage_budget_N']
    print(f"  - Dataset Size Reduction: {storage_budget}/{full_dataset_size} = {storage_budget/full_dataset_size:.4f}")
    
    # Additional analysis
    dataset_compression = full_dataset_size / storage_budget
    efficiency_score = accuracy_ratio / time_ratio
    
    print(f"\nAdditional Analysis:")
    print(f"  - Dataset Compression Ratio: {dataset_compression:.1f}x")
    print(f"  - Efficiency Score (Accuracy/Time): {efficiency_score:.4f}")
    
    if accuracy_ratio > 0.8:
        print(f"  - Result: EXCELLENT! Distilled dataset retains >80% of performance")
    elif accuracy_ratio > 0.6:
        print(f"  - Result: GOOD! Distilled dataset retains >60% of performance")
    else:
        print(f"  - Result: Distilled dataset performance could be improved")
    
    print("\n" + "="*80)
    print("Evaluation completed successfully!")
    print("="*80)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate distilled datasets.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file (e.g., configs/stanford_dogs.yaml)')
    
    args = parser.parse_args()

    evaluate_models(args.config)
