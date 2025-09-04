import os
import yaml
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

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

def pretrain_on_full_dataset(config, save_path=None):
    """Train a ResNet18 feature extractor on the full CIFAR100 dataset using MSE loss."""
    print("=" * 60)
    print("TRAINING RESNET18 ON FULL CIFAR100 DATASET")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, _ = get_dataset(config['data']['name'])
    
    # Load teacher model to get target representations
    teacher_model = get_teacher_model(
        config['models']['teacher']['path'], 
        config['models']['teacher']['feature_dim']
    ).to(device)
    
    # Create ResNet18 model
    model = ResNet18FeatureExtractor(feature_dim=512).to(device)
    
    # Training setup
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Training loop
    epochs = 1000
    print(f"Training for {epochs} epochs...")
    
    model.train()
    for epoch in tqdm(range(epochs), desc="Training on full dataset"):
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
    """Train a ResNet18 feature extractor on the distilled dataset."""
    print("=" * 60)
    print("TRAINING RESNET18 ON DISTILLED DATASET")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load distilled data
    distilled_loader = DistilledDatasetLoader(config, asset_dir)
    distilled_images, distilled_representations = distilled_loader.get_distilled_data()
    
    print(f"Distilled dataset size: {len(distilled_images)} images")
    print(f"Image shape: {distilled_images.shape}")
    print(f"Representation shape: {distilled_representations.shape}")
    
    # Create ResNet18 model
    model = ResNet18FeatureExtractor(feature_dim=512).to(device)
    
    # Training setup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Training loop
    epochs = 1000
    print(f"Training for {epochs} epochs...")
    
    model.train()
    for epoch in tqdm(range(epochs), desc="Training on distilled dataset"):
        # Forward pass
        pred_repr = model.get_features(distilled_images)
        loss = criterion(pred_repr, distilled_representations)
        
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
    """Perform linear evaluation on CIFAR100."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Get datasets
    train_dataset, test_dataset = get_dataset(config['data']['name'])
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Always extract train features for training the classifier
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=4, drop_last=False)
    train_features = []
    train_labels = []
    
    print("Extracting train features for classifier training...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Extracting train features")):
            images = images.to(device)
            features = model.get_features(images)
            
            # Debug: Check batch sizes
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}: images {images.shape}, features {features.shape}, labels {labels.shape}")
            
            train_features.append(features.cpu().numpy())
            train_labels.append(labels.numpy())
    
    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    
    print(f"Final train features shape: {train_features.shape}, Train labels shape: {train_labels.shape}")
    
    # Check for any NaN or inf values
    if np.any(np.isnan(train_features)) or np.any(np.isinf(train_features)):
        print("WARNING: NaN or Inf values found in train features!")
        train_features = np.nan_to_num(train_features)
    
    # Extract test features for evaluation
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4, drop_last=False)
    test_features = []
    test_labels = []
    
    print("Extracting test features for evaluation...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Extracting test features")):
            images = images.to(device)
            features = model.get_features(images)
            
            # Debug: Check batch sizes
            if batch_idx % 20 == 0:
                print(f"Test Batch {batch_idx}: images {images.shape}, features {features.shape}, labels {labels.shape}")
            
            test_features.append(features.cpu().numpy())
            test_labels.append(labels.numpy())
    
    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    print(f"Final test features shape: {test_features.shape}, Test labels shape: {test_labels.shape}")
    
    # Check for any NaN or inf values
    if np.any(np.isnan(test_features)) or np.any(np.isinf(test_features)):
        print("WARNING: NaN or Inf values found in test features!")
        test_features = np.nan_to_num(test_features)
    
    # Verify shapes before training classifier
    assert train_features.shape[0] == train_labels.shape[0], f"Train feature/label mismatch: {train_features.shape[0]} vs {train_labels.shape[0]}"
    assert test_features.shape[0] == test_labels.shape[0], f"Test feature/label mismatch: {test_features.shape[0]} vs {test_labels.shape[0]}"
    
    # Train linear classifier on train features
    print("Training linear classifier...")
    try:
        classifier = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        classifier.fit(train_features, train_labels)
        
        # Evaluate on test features
        predictions = classifier.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        
        print(f"Linear Evaluation Accuracy: {accuracy:.4f}")
        return accuracy
        
    except Exception as e:
        print(f"Error in classifier training: {e}")
        print(f"Train features shape: {train_features.shape}")
        print(f"Train labels shape: {train_labels.shape}")
        raise

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
    print("EXPERIMENT 1: ResNet18 trained on FULL CIFAR100 dataset")
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
    print("EXPERIMENT 2: ResNet18 trained on DISTILLED dataset")
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
    print("EVALUATION RESULTS COMPARISON")
    print("="*80)
    
    print(f"Full Dataset Training:")
    print(f"  - Test Accuracy: {results['full_dataset']['accuracy']:.4f}")
    print(f"  - Training Time: {results['full_dataset']['training_time']:.2f} seconds")
    
    print(f"\nDistilled Dataset Training:")
    print(f"  - Test Accuracy: {results['distilled_dataset']['accuracy']:.4f}")
    print(f"  - Training Time: {results['distilled_dataset']['training_time']:.2f} seconds")
    
    # Calculate efficiency metrics
    accuracy_ratio = results['distilled_dataset']['accuracy'] / results['full_dataset']['accuracy']
    time_ratio = results['distilled_dataset']['training_time'] / results['full_dataset']['training_time']
    
    print(f"\nEfficiency Metrics:")
    print(f"  - Accuracy Retention: {accuracy_ratio:.4f} ({accuracy_ratio*100:.2f}%)")
    print(f"  - Training Time Ratio: {time_ratio:.4f} ({time_ratio*100:.2f}%)")
    print(f"  - Dataset Size Reduction: {config['distillation']['storage_budget_N']}/50000 = {config['distillation']['storage_budget_N']/50000:.4f}")
    
    # Additional analysis
    dataset_compression = 50000 / config['distillation']['storage_budget_N']
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
    config_path = 'configs/cifar100.yaml'
    results = evaluate_models(config_path)
