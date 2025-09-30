import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm

from utils import get_dataset

class BarlowTwinsTransform:
    def __init__(self, size=32, mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]):
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=size,scale=(0.2,1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.2,hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p = 0.5),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def __call__(self,x):
        return self.transforms(x), self.transforms(x)

class Projector(nn.Module):
    def __init__(self,in_dim,hidden_dim=2048,out_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim)
        )
    
    def forward(self,x):
        return self.net(x)

class BarlowTwinsLoss(nn.Module):
    def __init__(self,lambda_coeff=5e-3):
        super().__init__()
        self.lambda_coeff = lambda_coeff
    
    def forward(self,z1,z2):
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)

        batch_size = z2.size(0)

        c = (z1_norm.T @ z2_norm) / batch_size
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = c.fill_diagonal_(0).pow_(2).sum()

        loss = on_diag + self.lambda_coeff*off_diag
        return loss

def main():
    dataset_name = "CIFAR100"
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1].upper()

    print("Starting Teacher Model Pre-training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if dataset_name == "CIFAR100":
        epochs = 100
        batch_size = 256
        learning_rate = 1e-3
        size = 32
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        save_path = os.path.join("./teacher_models", "resnet18_barlow_twins_cifar100.pth")
    elif dataset_name == "STANFORD_DOGS":
        epochs = 200  # More epochs for higher resolution and fine-grained dataset
        batch_size = 128  # Smaller batch size for 64x64 images as per paper's approach
        learning_rate = 1e-3  # Consistent with paper's approach across datasets
        size = 64
        # ImageNet normalization - paper uses this for transfer learning consistency
        # This is standard practice for fine-tuning and transfer learning scenarios
        mean = [0.485, 0.456, 0.406]  # ImageNet normalization values
        std = [0.229, 0.224, 0.225]   # ImageNet normalization values
        save_path = os.path.join("./teacher_models", "resnet18_barlow_twins_stanford_dogs.pth")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    save_dir = "./teacher_models"
    os.makedirs(save_dir, exist_ok=True)

    train_dataset, _ = get_dataset(dataset_name)
    train_dataset.transform = BarlowTwinsTransform(size=size, mean=mean, std=std)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    backbone = resnet18()
    backbone.fc = nn.Identity()

    projector = Projector(in_dim=512)

    model = nn.Sequential(backbone,projector).to(device)

    loss_fn = BarlowTwinsLoss().to(device)
    optimizer = optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-5)

    print("Starting Training Loop....")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader,desc=f"Epoch [{epoch+1}/{epochs}]",leave=False)

        for (view1,view2), _ in loop:
            view1,view2 = view1.to(device),view2.to(device)

            z1 = model(view1)
            z2 = model(view2)

            loss = loss_fn(z1,z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss/len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_loss:.4f}")

    print("Training Complete. Saving Model....")
    torch.save(backbone.state_dict(),save_path)
    print(f"Teacher model backbone saved to {save_path}")

if __name__ == '__main__':
    main()