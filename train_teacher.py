import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
import yaml

from utils import get_dataset

class BarlowTwins(nn.Module):
    def __init__(self, backbone, projection_dim=128, hidden_dim=512):
        super().__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()
        
        self.projector = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return features, projections

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_coeff=5e-3):
        super().__init__()
        self.lambda_coeff = lambda_coeff
        
    def forward(self, z1, z2):
        batch_size = z1.size(0)
        feature_dim = z1.size(1)
        
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        cross_corr = torch.mm(z1_norm.T, z2_norm) / batch_size
        
        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = cross_corr.flatten()[1:].view(feature_dim-1, feature_dim+1)[:, :-1].pow_(2).sum()
        
        loss = on_diag + self.lambda_coeff * off_diag
        return loss

def get_augmentation_pipeline():
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

def train_teacher_model():
    print("="*60)
    print("TRAINING TEACHER MODEL USING BARLOW TWINS")
    print("="*60)
    
    total_start_time = time.time()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(device))
    
    print("\n⏱️  Setting up data pipeline...")
    setup_start = time.time()
    transform = get_augmentation_pipeline()
    
    train_dataset, _ = get_dataset('CIFAR100')
    train_dataset.transform = transform
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, drop_last=True)
    setup_time = time.time() - setup_start
    print("✓ Data setup completed in {:.2f}s".format(setup_time))
    
    print("\n⏱️  Initializing model...")
    model_start = time.time()
    backbone = resnet18()
    model = BarlowTwins(backbone, projection_dim=128, hidden_dim=512).to(device)
    
    criterion = BarlowTwinsLoss(lambda_coeff=5e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800)
    model_setup_time = time.time() - model_start
    print("✓ Model initialization completed in {:.2f}s".format(model_setup_time))
    
    epochs = 800
    model.train()
    
    print("\n⏱️  Starting training for {} epochs...".format(epochs))
    training_start = time.time()
    
    epoch_times = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        num_batches = 0
        
        batch_times = []
        for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc="Epoch {}/{}".format(epoch+1, epochs), leave=False)):
            batch_start = time.time()
            
            images = images.to(device)
            
            aug1 = images
            aug2 = torch.stack([transform(transforms.ToPILImage()(img)) for img in images.cpu()])
            aug2 = aug2.to(device)
            
            _, z1 = model(aug1)
            _, z2 = model(aug2)
            
            loss = criterion(z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if batch_idx == 0 and epoch == 0:
                print("  📊 First batch processed in {:.2f}s".format(batch_time))
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        avg_batch_time = sum(batch_times) / len(batch_times)
        
        if (epoch + 1) % 50 == 0:
            elapsed_training = time.time() - training_start
            remaining_epochs = epochs - (epoch + 1)
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            eta = remaining_epochs * avg_epoch_time
            
            print("Epoch {}/{}".format(epoch+1, epochs))
            print("  Loss: {:.6f} (Best: {:.6f})".format(avg_loss, best_loss))
            print("  LR: {:.6f}".format(scheduler.get_last_lr()[0]))
            print("  Epoch Time: {:.1f}s (Avg: {:.1f}s)".format(epoch_time, avg_epoch_time))
            print("  Batch Time: {:.3f}s".format(avg_batch_time))
            print("  Elapsed: {:.1f}h, ETA: {:.1f}h".format(elapsed_training/3600, eta/3600))
    
    training_time = time.time() - training_start
    print("\n✓ Training completed in {:.1f} hours".format(training_time/3600))
    
    print("\n⏱️  Saving model...")
    save_start = time.time()
    os.makedirs('./teacher_models', exist_ok=True)
    save_path = './teacher_models/resnet18_barlow_twins_cifar100.pth'
    torch.save(model.backbone.state_dict(), save_path)
    save_time = time.time() - save_start
    print("✓ Model saved in {:.2f}s to {}".format(save_time, save_path))
    
    total_time = time.time() - total_start_time
    print("\n{}".format("="*60))
    print("TEACHER TRAINING SUMMARY")
    print("{}".format("="*60))
    print("Total Time: {:.1f} hours".format(total_time/3600))
    print("  Setup: {:.1f}s".format(setup_time + model_setup_time))
    print("  Training: {:.1f}h".format(training_time/3600))
    print("  Saving: {:.1f}s".format(save_time))
    print("Final Loss: {:.6f}".format(best_loss))
    print("Average Epoch Time: {:.1f}s".format(sum(epoch_times)/len(epoch_times)))
    print("{}".format("="*60))
    
    return model.backbone

if __name__ == "__main__":
    train_teacher_model()
