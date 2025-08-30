import os
import yaml
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm

from utils import get_dataset
from models import DistilledData, ApproximationMLP
from main_distill import DistilledData

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = resnet18()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class DistilledDatasetLoader:
    """Helper class to load and reconstruct distilled dataset."""
    def __init__(self, config, asset_dir):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load distilled data
        self.distilled_data = self._load_distilled_data(asset_dir)
        
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
    
    def get_distilled_data_with_labels(self):
        """Get the distilled images with pseudo-labels."""
        with torch.no_grad():
            images = self.distilled_data.reconstruct_images()
        
        # Create pseudo-labels (distribute evenly across classes)
        num_images = len(images)
        num_classes = 100  # CIFAR100 has 100 classes
        labels = torch.arange(num_classes).repeat(num_images // num_classes + 1)[:num_images]
        
        return images, labels

def train_on_full_dataset(config, save_path=None):
    """Train ResNet18 classifier on full CIFAR100 dataset."""
    print("=" * 60)
    print("TRAINING RESNET18 CLASSIFIER ON FULL CIFAR100 DATASET")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get datasets
    train_dataset, test_dataset = get_dataset(config['data']['name'])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # Create model
    model = ResNet18Classifier(num_classes=100).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    
    epochs = 200
    best_acc = 0.0
    
    print(f"Training for {epochs} epochs...")
    print(f"Dataset size: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        # Evaluation phase
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            train_acc = 100.0 * train_correct / train_total
            test_acc = 100.0 * test_correct / test_total
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Acc: {test_acc:.2f}%")
            
            if test_acc > best_acc:
                best_acc = test_acc
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(model.state_dict(), save_path)
    
    print(f"Training completed. Best test accuracy: {best_acc:.2f}%")
    return model, best_acc

def evaluate_model(model, config):
    """Evaluate model on CIFAR100 test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    _, test_dataset = get_dataset(config['data']['name'])
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy

def train_on_distilled_dataset(config, asset_dir, save_path=None):
    """Train ResNet18 classifier on distilled dataset using data augmentation."""
    print("=" * 60)
    print("TRAINING RESNET18 CLASSIFIER ON DISTILLED DATASET")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load distilled data
    distilled_loader = DistilledDatasetLoader(config, asset_dir)
    distilled_images, distilled_labels = distilled_loader.get_distilled_data_with_labels()
    
    print(f"Distilled dataset size: {len(distilled_images)} images")
    print(f"Image shape: {distilled_images.shape}")
    
    # Create model
    model = ResNet18Classifier(num_classes=100).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    epochs = 1000  # More epochs since we have less data
    best_acc = 0.0
    
    print(f"Training for {epochs} epochs...")
    
    # Get test dataset for evaluation
    _, test_dataset = get_dataset(config['data']['name'])
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    for epoch in range(epochs):
        model.train()
        
        # Train on distilled data
        optimizer.zero_grad()
        outputs = model(distilled_images)
        loss = criterion(outputs, distilled_labels.to(device))
        loss.backward()
        optimizer.step()
        
        # Evaluate every 100 epochs
        if (epoch + 1) % 100 == 0:
            model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            test_acc = 100.0 * test_correct / test_total
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {loss.item():.4f}")
            print(f"  Test Acc: {test_acc:.2f}%")
            
            if test_acc > best_acc:
                best_acc = test_acc
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(model.state_dict(), save_path)
    
    print(f"Training completed. Best test accuracy: {best_acc:.2f}%")
    return model, best_acc

def compare_classification_performance(config_path):
    """Compare classification performance: full dataset vs distilled dataset."""
    print("Starting Classification Performance Comparison...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    asset_dir = config['saving']['distilled_assets_dir']
    
    # Check if distilled assets exist
    if not os.path.exists(asset_dir):
        print(f"Error: Distilled assets not found at {asset_dir}")
        print("Please run main_distill.py first to generate distilled dataset.")
        return
    
    # Create save directories
    eval_dir = "./classification_models"
    full_model_path = os.path.join(eval_dir, "resnet18_classifier_full.pth")
    distilled_model_path = os.path.join(eval_dir, "resnet18_classifier_distilled.pth")
    
    results = {}
    
    # Train on full dataset
    print("\n" + "="*80)
    print("EXPERIMENT 1: ResNet18 Classification on FULL CIFAR100")
    print("="*80)
    
    start_time = time.time()
    full_model, full_best_acc = train_on_full_dataset(config, save_path=full_model_path)
    full_training_time = time.time() - start_time
    
    # Final evaluation on full dataset model
    final_full_acc = evaluate_model(full_model, config)
    
    results['full_dataset'] = {
        'best_accuracy': full_best_acc,
        'final_accuracy': final_full_acc,
        'training_time': full_training_time
    }
    
    # Train on distilled dataset
    print("\n" + "="*80)
    print("EXPERIMENT 2: ResNet18 Classification on DISTILLED DATASET")
    print("="*80)
    
    start_time = time.time()
    distilled_model, distilled_best_acc = train_on_distilled_dataset(
        config, asset_dir, save_path=distilled_model_path
    )
    distilled_training_time = time.time() - start_time
    
    # Final evaluation on distilled dataset model
    final_distilled_acc = evaluate_model(distilled_model, config)
    
    results['distilled_dataset'] = {
        'best_accuracy': distilled_best_acc,
        'final_accuracy': final_distilled_acc,
        'training_time': distilled_training_time
    }
    
    # Print comparison results
    print("\n" + "="*80)
    print("CLASSIFICATION PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"Full Dataset Training:")
    print(f"  - Best Test Accuracy: {results['full_dataset']['best_accuracy']:.2f}%")
    print(f"  - Final Test Accuracy: {results['full_dataset']['final_accuracy']:.2f}%")
    print(f"  - Training Time: {results['full_dataset']['training_time']:.2f} seconds")
    
    print(f"\nDistilled Dataset Training:")
    print(f"  - Best Test Accuracy: {results['distilled_dataset']['best_accuracy']:.2f}%")
    print(f"  - Final Test Accuracy: {results['distilled_dataset']['final_accuracy']:.2f}%")
    print(f"  - Training Time: {results['distilled_dataset']['training_time']:.2f} seconds")
    
    # Calculate efficiency metrics
    accuracy_ratio = results['distilled_dataset']['final_accuracy'] / results['full_dataset']['final_accuracy']
    time_ratio = results['distilled_dataset']['training_time'] / results['full_dataset']['training_time']
    
    print(f"\nEfficiency Metrics:")
    print(f"  - Accuracy Retention: {accuracy_ratio:.4f} ({accuracy_ratio*100:.2f}%)")
    print(f"  - Training Time Ratio: {time_ratio:.4f} ({time_ratio*100:.2f}%)")
    print(f"  - Dataset Size Reduction: {config['distillation']['storage_budget_N']}/50000 = {config['distillation']['storage_budget_N']/50000:.4f}")
    
    dataset_compression = 50000 / config['distillation']['storage_budget_N']
    efficiency_score = accuracy_ratio / time_ratio
    
    print(f"\nAdditional Analysis:")
    print(f"  - Dataset Compression Ratio: {dataset_compression:.1f}x")
    print(f"  - Efficiency Score (Accuracy/Time): {efficiency_score:.4f}")
    
    print("\n" + "="*80)
    print("Classification comparison completed!")
    print("="*80)
    
    return results

if __name__ == "__main__":
    config_path = 'configs/cifar100.yaml'
    results = compare_classification_performance(config_path)
