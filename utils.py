import torch
import numpy as np
import os
from PIL import Image
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

class StanfordDogsDataset(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        
        # Path to the actual dataset
        self.images_dir = os.path.join(root, 'full_datasets', 'stanford-dogs', 'images', 'Images')
        
        if not os.path.exists(self.images_dir):
            raise RuntimeError(
                f"Stanford Dogs dataset not found at {self.images_dir}.\n"
                f"Please ensure the dataset is downloaded to: {root}/full_datasets/stanford-dogs/"
            )
        
        # Load all samples
        self.samples = []
        self.class_to_idx = {}
        
        # Get all breed directories (120 breeds)
        breed_dirs = sorted([d for d in os.listdir(self.images_dir) 
                           if os.path.isdir(os.path.join(self.images_dir, d)) and d.startswith('n0')])
        
        print(f"Found {len(breed_dirs)} dog breeds")
        
        for idx, breed in enumerate(breed_dirs):
            self.class_to_idx[breed] = idx
            breed_path = os.path.join(self.images_dir, breed)
            
            # Get all images in this breed directory
            images = [f for f in os.listdir(breed_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            # Create train/test split (80/20)
            images.sort()  # Ensure consistent ordering
            split_idx = int(0.8 * len(images))
            
            if train:
                breed_images = images[:split_idx]
            else:
                breed_images = images[split_idx:]
            
            # Add samples
            for img in breed_images:
                img_path = os.path.join(breed_path, img)
                self.samples.append((img_path, idx))
        
        self.classes = list(self.class_to_idx.keys())
        print(f"Loaded {len(self.samples)} {'training' if train else 'testing'} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
            
        return image, target

def get_dataset(name,data_dir='./data'):
    if name.upper() == "CIFAR100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761))
        ])

        train_dataset = datasets.CIFAR100(root=data_dir,train=True,download=True,
        transform=transform)

        test_dataset = datasets.CIFAR100(root=data_dir,train=False,download=True,
        transform=transform)

        return train_dataset, test_dataset
    elif name.upper() == "STANFORD_DOGS":
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        train_dataset = StanfordDogsDataset(root=data_dir,train=True,download=True,
        transform=transform)

        test_dataset = StanfordDogsDataset(root=data_dir,train=False,download=True,
        transform=transform)

        return train_dataset, test_dataset
    else:
        raise NotImplementedError(f"Dataset '{name}' is not supported.")

def get_pca_components(data,n_components):
    print(f"Performing PCA to find {n_components} components....")

    if data.ndim > 2:
        data = data.reshape(data.shape[0],-1)
    
    pca = PCA(n_components=n_components,random_state=42)
    pca.fit(data)

    print("PCA fitting complete")
    return pca.components_

def extract_all_data(dataset):
    loader = DataLoader(dataset,batch_size=1024,shuffle=False,num_workers=4)
    all_images = []
    print("Extracting all images from the specified dataset.....")
    for images,_ in loader:
        all_images.append(images.numpy())
    full_dataset_array = np.concatenate(all_images,axis=0)
    print("Data Extraction complete")
    return full_dataset_array

if __name__ == '__main__':
    cifar100_train, _ = get_dataset('CIFAR100')
    print("Loaded CIFAR100")

    all_images = extract_all_data(cifar100_train)
    n_components = 64
    bases = get_pca_components(all_images,n_components)

    print(f"Sucessful. Shape of image bases : {bases.shape}")