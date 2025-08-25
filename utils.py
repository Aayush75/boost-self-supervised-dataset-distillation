import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

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