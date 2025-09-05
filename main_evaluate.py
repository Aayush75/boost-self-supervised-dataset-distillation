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

class AugmentationStrategy:
    def __init__(self, config):
        self.config = config
        
    def apply_augmentation(self, images, aug_type, strength=None):
        if aug_type == 'rotate':
            angle = strength if strength is not None else np.random.choice(self.config['augmentations']['rotate'])
            return transforms.functional.rotate(images, float(angle))
        elif aug_type == 'color_jitter':
            brightness = strength if strength is not None else self.config['augmentations']['color_jitter']['brightness']
            contrast = strength if strength is not None else self.config['augmentations']['color_jitter']['contrast']
            saturation = strength if strength is not None else self.config['augmentations']['color_jitter']['saturation']
            hue = strength/4 if strength is not None else self.config['augmentations']['color_jitter']['hue']
            transform = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
            return transform(images)
        elif aug_type == 'random_crop':
            size = images.shape[-1]
            scale = (strength, 1.0) if strength is not None else self.config['augmentations']['random_crop']['scale']
            ratio = self.config['augmentations']['random_crop']['ratio']
            transform = transforms.RandomResizedCrop(size, scale=scale, ratio=ratio)
            return transform(images)
        elif aug_type == 'horizontal_flip':
            return transforms.functional.hflip(images)
        elif aug_type == 'gaussian_blur':
            kernel_size = int(strength) if strength is not None else np.random.choice(self.config['augmentations']['gaussian_blur']['kernel_size'])
            sigma = np.random.uniform(*self.config['augmentations']['gaussian_blur']['sigma'])
            return transforms.functional.gaussian_blur(images, kernel_size, sigma)
        return images

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
        approx_networks = {}
        
        for rot_angle in self.config['augmentations']['rotate']:
            net = ApproximationMLP(
                num_repr_bases_V=self.config['parametrization']['repr_bases_V'],
                hidden_dim=self.config['models']['approximation_mlp']['hidden_dim']
            ).to(self.device)
            net_path = os.path.join(asset_dir, f'approx_net_rot_{rot_angle}.pth')
            if os.path.exists(net_path):
                net.load_state_dict(torch.load(net_path, map_location=self.device))
                net.eval()
            approx_networks[f'rotate_{rot_angle}'] = net
        
        if 'color_jitter_strengths' in self.config['augmentations']:
            for strength in self.config['augmentations']['color_jitter_strengths']:
                net = ApproximationMLP(
                    num_repr_bases_V=self.config['parametrization']['repr_bases_V'],
                    hidden_dim=self.config['models']['approximation_mlp']['hidden_dim']
                ).to(self.device)
                net_path = os.path.join(asset_dir, f'approx_net_color_{strength}.pth')
                if os.path.exists(net_path):
                    net.load_state_dict(torch.load(net_path, map_location=self.device))
                    net.eval()
                approx_networks[f'color_jitter_{strength}'] = net
        
        if 'crop_scales' in self.config['augmentations']:
            for scale in self.config['augmentations']['crop_scales']:
                net = ApproximationMLP(
                    num_repr_bases_V=self.config['parametrization']['repr_bases_V'],
                    hidden_dim=self.config['models']['approximation_mlp']['hidden_dim']
                ).to(self.device)
                net_path = os.path.join(asset_dir, f'approx_net_crop_{scale}.pth')
                if os.path.exists(net_path):
                    net.load_state_dict(torch.load(net_path, map_location=self.device))
                    net.eval()
                approx_networks[f'crop_{scale}'] = net
        
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
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], momentum=config['training']['momentum'], weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['max_epochs'])
    criterion = nn.MSELoss()
    
    # Training loop
    epochs = config['training']['max_epochs']
    print(f"Training for {epochs} epochs...")
    
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
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
        
        avg_loss = total_loss / num_batches
        scheduler.step()
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config['training']['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
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
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], momentum=config['training']['momentum'], weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['max_epochs'])
    criterion = nn.MSELoss()
    
    # Training loop
    epochs = config['training']['max_epochs']
    print(f"Training for {epochs} epochs...")
    
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(epochs), desc="Training on distilled dataset"):
        # Forward pass
        pred_repr = model.get_features(distilled_images)
        loss = criterion(pred_repr, distilled_representations)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        scheduler.step()
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config['training']['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
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
    train_dataset, test_dataset = get_dataset(config['data']['name'])
    print(f"Dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # Create linear classifier (only this will be trained)
    feature_dim = 512  # ResNet18 feature dimension
    num_classes = 100  # CIFAR100 has 100 classes
    linear_classifier = nn.Linear(feature_dim, num_classes).to(device)
    
    # Training setup for linear classifier only
    optimizer = torch.optim.SGD(linear_classifier.parameters(), lr=config['training']['linear_eval_lr'], momentum=config['training']['momentum'], weight_decay=config['training']['linear_eval_weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['linear_eval_epochs'])
    criterion = nn.CrossEntropyLoss()
    
    # Training loop for linear classifier
    epochs = config['training']['linear_eval_epochs']
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

def compute_additional_metrics(model, teacher_model, config):
    """Compute additional evaluation metrics from the paper"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, test_dataset = get_dataset(config['data']['name'])
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    model.eval()
    teacher_model.eval()
    
    def representation_similarity():
        cosine_similarities = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                student_repr = model.get_features(images)
                teacher_repr = teacher_model(images)
                sim = F.cosine_similarity(student_repr, teacher_repr, dim=1)
                cosine_similarities.extend(sim.cpu().numpy())
        return np.mean(cosine_similarities)
    
    def per_class_accuracy():
        class_correct = torch.zeros(config['data']['num_classes'])
        class_total = torch.zeros(config['data']['num_classes'])
        
        # Create a linear classifier for per-class evaluation
        linear_classifier = nn.Linear(512, config['data']['num_classes']).to(device)
        optimizer = torch.optim.SGD(linear_classifier.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        
        # Quick training of linear classifier
        train_dataset, _ = get_dataset(config['data']['name'])
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
        
        for epoch in range(10):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    features = model.get_features(images)
                outputs = linear_classifier(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate per class
        linear_classifier.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                features = model.get_features(images)
                outputs = linear_classifier(features)
                _, predicted = torch.max(outputs, 1)
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        class_accuracies = (class_correct / (class_total + 1e-8)).numpy()
        return class_accuracies
    
    def compute_efficiency_metrics():
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024
        
        full_dataset_size = 50000 * 3 * 32 * 32 * 4
        distilled_dataset_size = config['distillation']['storage_budget_N'] * 3 * 32 * 32 * 4
        compression_ratio = full_dataset_size / distilled_dataset_size
        
        return {
            'memory_usage_mb': memory_usage,
            'compression_ratio': compression_ratio,
            'storage_reduction': 1 - (distilled_dataset_size / full_dataset_size)
        }
    
    repr_sim = representation_similarity()
    class_accs = per_class_accuracy()
    efficiency = compute_efficiency_metrics()
    
    return {
        'representation_similarity': repr_sim,
        'per_class_accuracy': class_accs,
        'worst_class_accuracy': np.min(class_accs),
        'best_class_accuracy': np.max(class_accs),
        'class_accuracy_std': np.std(class_accs),
        'memory_usage_mb': efficiency['memory_usage_mb'],
        'compression_ratio': efficiency['compression_ratio'],
        'storage_reduction': efficiency['storage_reduction']
    }

def evaluate_models(config_path):
    """Main evaluation function comparing full dataset vs distilled dataset training."""
    print("="*80)
    print("MODEL EVALUATION PIPELINE")
    print("="*80)
    
    total_start_time = time.time()
    
    # Load config
    print("\n⏱️  Loading configuration...")
    config_start = time.time()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config_time = time.time() - config_start
    print(f"✓ Configuration loaded in {config_time:.2f}s")
    
    asset_dir = config['saving']['distilled_assets_dir']
    
    # Check if distilled assets exist
    print(f"\n⏱️  Checking distilled assets...")
    check_start = time.time()
    if not os.path.exists(asset_dir):
        print(f"Error: Distilled assets not found at {asset_dir}")
        print("Please run main_distill.py first to generate distilled dataset.")
        return
    check_time = time.time() - check_start
    print(f"✓ Assets verified in {check_time:.2f}s")
    
    # Create save directories for models
    eval_dir = "./evaluation_models"
    full_model_path = os.path.join(eval_dir, "resnet18_full_dataset.pth")
    distilled_model_path = os.path.join(eval_dir, "resnet18_distilled_dataset.pth")
    
    results = {}
    
    # Train and evaluate model on full dataset
    print("\n" + "="*80)
    print("EXPERIMENT 1: ResNet18 trained on FULL CIFAR100 dataset")
    print("="*80)
    
    full_start_time = time.time()
    full_model = pretrain_on_full_dataset(config, save_path=full_model_path)
    full_training_time = time.time() - full_start_time
    print(f"⏱️  Full dataset training completed in {full_training_time/3600:.1f}h")
    
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
    
    distilled_start_time = time.time()
    distilled_model = pretrain_on_distilled_dataset(config, asset_dir, save_path=distilled_model_path)
    distilled_training_time = time.time() - distilled_start_time
    print(f"⏱️  Distilled dataset training completed in {distilled_training_time/3600:.1f}h")
    
    print("\n⏱️  Evaluating distilled dataset model...")
    eval_start = time.time()
    distilled_accuracy = linear_evaluation(distilled_model, config, split='test')
    eval_time = time.time() - eval_start
    print(f"✓ Distilled model evaluation completed in {eval_time:.2f}s")
    
    results['distilled_dataset'] = {
        'accuracy': distilled_accuracy,
        'training_time': distilled_training_time
    }
    
    # Compute additional metrics
    print("\n⏱️  Computing additional evaluation metrics...")
    metrics_start = time.time()
    teacher_model = get_teacher_model(
        config['models']['teacher']['path'], 
        config['models']['teacher']['feature_dim']
    )
    
    full_additional_metrics = compute_additional_metrics(full_model, teacher_model, config)
    distilled_additional_metrics = compute_additional_metrics(distilled_model, teacher_model, config)
    metrics_time = time.time() - metrics_start
    print(f"✓ Additional metrics computed in {metrics_time:.2f}s")
    
    results['full_dataset']['additional_metrics'] = full_additional_metrics
    results['distilled_dataset']['additional_metrics'] = distilled_additional_metrics
    
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
    
    # Memory analysis
    full_dataset_size = 50000 * 3 * 32 * 32 * 4  # bytes (float32)
    distilled_dataset_size = config['distillation']['storage_budget_N'] * 3 * 32 * 32 * 4
    memory_ratio = distilled_dataset_size / full_dataset_size
    print(f"  - Memory Usage Ratio: {memory_ratio:.4f} ({memory_ratio*100:.2f}%)")
    
    # Performance per computation cost
    efficiency_metric = accuracy_ratio * dataset_compression
    print(f"  - Performance-Compression Product: {efficiency_metric:.4f}")
    
    if accuracy_ratio > 0.8:
        print(f"  - Result: EXCELLENT! Distilled dataset retains >80% of performance with {dataset_compression:.1f}x compression")
    elif accuracy_ratio > 0.6:
        print(f"  - Result: GOOD! Distilled dataset retains >60% of performance with {dataset_compression:.1f}x compression")
    else:
        print(f"  - Result: Distilled dataset performance could be improved (only {accuracy_ratio*100:.1f}% retention)")
        
    # Statistical significance indicators
    print(f"\nStatistical Analysis:")
    print(f"  - Absolute Accuracy Drop: {(results['full_dataset']['accuracy'] - results['distilled_dataset']['accuracy'])*100:.2f} percentage points")
    print(f"  - Relative Performance: {accuracy_ratio:.3f} ({accuracy_ratio*100:.1f}% of original)")
    print(f"  - Training Speedup: {1/time_ratio:.2f}x faster")
    
    # Additional metrics reporting
    print(f"\nRepresentation Quality Analysis:")
    print(f"  Full Dataset - Teacher Similarity: {full_additional_metrics['representation_similarity']:.4f}")
    print(f"  Distilled Dataset - Teacher Similarity: {distilled_additional_metrics['representation_similarity']:.4f}")
    
    print(f"\nPer-Class Performance Analysis:")
    print(f"  Full Dataset - Worst Class: {full_additional_metrics['worst_class_accuracy']:.4f}")
    print(f"  Full Dataset - Best Class: {full_additional_metrics['best_class_accuracy']:.4f}")
    print(f"  Full Dataset - Std Dev: {full_additional_metrics['class_accuracy_std']:.4f}")
    print(f"  Distilled Dataset - Worst Class: {distilled_additional_metrics['worst_class_accuracy']:.4f}")
    print(f"  Distilled Dataset - Best Class: {distilled_additional_metrics['best_class_accuracy']:.4f}")
    print(f"  Distilled Dataset - Std Dev: {distilled_additional_metrics['class_accuracy_std']:.4f}")
    
    print(f"\nComputational Efficiency:")
    print(f"  Memory Usage - Full: {full_additional_metrics['memory_usage_mb']:.1f} MB")
    print(f"  Memory Usage - Distilled: {distilled_additional_metrics['memory_usage_mb']:.1f} MB")
    print(f"  Storage Compression: {distilled_additional_metrics['compression_ratio']:.1f}x")
    print(f"  Storage Reduction: {distilled_additional_metrics['storage_reduction']*100:.1f}%")
    
    if time_ratio < 0.1:
        print(f"  - Training Efficiency: EXCELLENT (>10x speedup)")
    elif time_ratio < 0.3:
        print(f"  - Training Efficiency: GOOD (>3x speedup)")
    else:
        print(f"  - Training Efficiency: MODERATE (<3x speedup)")
    
    total_eval_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"EVALUATION TIMING SUMMARY")
    print(f"{'='*80}")
    print(f"Total Evaluation Time: {total_eval_time/3600:.1f} hours")
    print(f"  Configuration & Setup: {(config_time + check_time):.1f}s")
    print(f"  Full Dataset Training: {full_training_time/3600:.1f}h")
    print(f"  Distilled Dataset Training: {distilled_training_time/3600:.1f}h")
    print(f"  Evaluation & Metrics: {(eval_time + metrics_time):.1f}s")
    print(f"Time Savings: {((full_training_time - distilled_training_time)/3600):.1f}h ({time_ratio*100:.1f}% of original time)")
    print(f"{'='*80}")
    
    print("\n" + "="*80)
    print("Evaluation completed successfully!")
    print("="*80)
    print("="*80)
    
    return results

if __name__ == "__main__":
    config_path = 'configs/cifar100.yaml'
    results = evaluate_models(config_path)
