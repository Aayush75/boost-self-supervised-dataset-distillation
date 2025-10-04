# Self-Supervised Dataset Distillation

Implementation of "Boost Self-Supervised Dataset Distillation via Parameterization, Predefined Augmentation, and Approximation" (arXiv:2507.21455v2).

## Overview

This method distills large datasets into compact, synthetic datasets while preserving performance through:
- **Parameterization**: Images and representations are represented via low-dimensional PCA bases
- **Predefined Augmentations**: Rotation augmentations (90°, 180°, 270°) to avoid gradient bias
- **Approximation Networks**: Small MLPs predict augmented representations to save storage

## Supported Datasets

- **CIFAR-10**: 32×32 RGB images, 10 classes, 50,000 training samples
- **CIFAR-100**: 32×32 RGB images, 100 classes, 50,000 training samples
- **Stanford Dogs**: 64×64 RGB images, 120 dog breeds, ~12,000 training samples

## Setup

### Requirements
```bash
pip install torch torchvision tqdm pyyaml scikit-learn pillow numpy
```

### Dataset Setup

#### CIFAR-10
Download the Python version from the [official CIFAR-10 page](https://www.cs.toronto.edu/~kriz/cifar.html) and extract it. Your directory structure should be:
```
./data/cifar-10-python/
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── test_batch
└── batches.meta
```

#### CIFAR-100
CIFAR-100 will be automatically downloaded by torchvision when you run the scripts. It will be saved to:
```
./data/data/cifar-100-python/
```

#### Stanford Dogs
Download from the [official dataset page](http://vision.stanford.edu/aditya86/ImageNetDogs/) or [Kaggle](https://www.kaggle.com/c/dog-breed-identification/data). Unzip and place the `Images` folder such that the structure is:
```
./data/stanford-dogs/
└── images/
    └── Images/
        ├── n02085620-Chihuahua/
        ├── n02085782-Japanese_spaniel/
        └── ... (120 breed folders)
```

## Usage

The distillation pipeline consists of three main steps: **Pre-training**, **Distillation**, and **Evaluation**.

### Complete Workflow

#### CIFAR-10

```bash
# Step 1: Pre-train teacher model using Barlow Twins (self-supervised learning)
python pretrain.py CIFAR10

# Step 2: Distill the dataset
python main_distill.py --config configs/cifar10.yaml

# Step 3: Evaluate the distilled dataset
python main_evaluate.py --config configs/cifar10.yaml
```

#### CIFAR-100

```bash
# Step 1: Pre-train teacher model
python pretrain.py CIFAR100

# Step 2: Distill the dataset
python main_distill.py --config configs/cifar100.yaml

# Step 3: Evaluate the distilled dataset
python main_evaluate.py --config configs/cifar100.yaml
```

#### Stanford Dogs

```bash
# Step 1: Pre-train teacher model
python pretrain.py STANFORD_DOGS

# Step 2: Distill the dataset
python main_distill.py --config configs/stanford_dogs.yaml

# Step 3: Evaluate the distilled dataset
python main_evaluate.py --config configs/stanford_dogs.yaml
```

### Testing Dataset Loading (Optional)

To verify that the Stanford Dogs dataset is correctly set up:
```bash
python test_stanford_dogs.py
```

## Configuration Files

All experiment settings are defined in YAML configuration files in the `configs/` directory.

### CIFAR-10 (`configs/cifar10.yaml`)
- **Storage budget**: 100 images (0.2% of original dataset)
- **Resolution**: 32×32
- **Image bases (U)**: 200
- **Representation bases (V)**: 200
- **Distillation steps**: 30,000
- **Evaluation epochs**: 1,000

### CIFAR-100 (`configs/cifar100.yaml`)
- **Storage budget**: 100 images (0.2% of original dataset)
- **Resolution**: 32×32
- **Image bases (U)**: 200
- **Representation bases (V)**: 200
- **Distillation steps**: 30,000
- **Evaluation epochs**: 1,000

### Stanford Dogs (`configs/stanford_dogs.yaml`)
- **Storage budget**: 240 images (2 per class, ~2% of original dataset)
- **Resolution**: 64×64
- **Image bases (U)**: 400
- **Representation bases (V)**: 400
- **Distillation steps**: 20,000
- **Evaluation epochs**: 1,000

You can modify these configuration files to experiment with different hyperparameters.

## Output Files

### Teacher Models (`teacher_models/`)
Pre-trained ResNet-18 backbones using Barlow Twins self-supervised learning:
- `resnet18_barlow_twins_cifar10.pth`
- `resnet18_barlow_twins_cifar100.pth`
- `resnet18_barlow_twins_stanford_dogs.pth`

### Distilled Assets (`distilled_assets/`)
Distilled datasets and approximation networks:
- `cifar10_N100/`
  - `distilled_data.pth` - Learned image and representation parameters
  - `approx_net_rot_90.pth`, `approx_net_rot_180.pth`, `approx_net_rot_270.pth` - Approximation MLPs
- `cifar100_N100/` - Same structure as CIFAR-10
- `stanford_dogs_N240/` - Same structure as CIFAR-10

### Evaluation Models (`evaluation_models/`)
Models trained on full vs distilled datasets for comparison:
- `resnet18_full_dataset.pth`
- `resnet18_distilled_dataset.pth`

## Repository Structure

```
boost-self-supervised-dataset-distillation/
├── configs/                    # YAML configuration files
│   ├── cifar10.yaml           # CIFAR-10 experiment settings
│   ├── cifar100.yaml          # CIFAR-100 experiment settings
│   └── stanford_dogs.yaml     # Stanford Dogs experiment settings
├── data/                      # Dataset directory (create this)
│   ├── cifar-10-python/       # CIFAR-10 batch files
│   ├── data/                  # CIFAR-100 (auto-downloaded)
│   └── stanford-dogs/         # Stanford Dogs images
├── teacher_models/            # Pre-trained teacher models (generated)
├── distilled_assets/          # Distilled datasets (generated)
├── evaluation_models/         # Evaluation models (generated)
├── main_distill.py           # Dataset distillation (bi-level optimization)
├── pretrain.py               # Teacher model pre-training (Barlow Twins)
├── main_evaluate.py          # Evaluation and comparison
├── run_evaluation.py         # Evaluation runner script
├── test_stanford_dogs.py     # Dataset loading test for Stanford Dogs
├── utils.py                  # Dataset loaders and utilities
├── models.py                 # Neural network architectures
└── README.md                 # This file
```

## Method Details

### 1. Teacher Pre-training
A ResNet-18 backbone is trained using **Barlow Twins**, a self-supervised learning method that learns rich representations without labels by maximizing agreement between augmented views of the same image.

### 2. Dataset Distillation
The distillation process uses a **bi-level optimization** framework:
- **Outer loop**: Optimizes the distilled data parameters (image coefficients `C_x`, representation coefficients `C_y`)
- **Inner loop**: Trains a pool of small CNNs on the current distilled data
- **Approximation networks**: Small MLPs learn to predict how representations change under augmentations

### 3. Evaluation
A new ResNet-18 is trained from scratch on the distilled dataset and evaluated via **linear probing** (training only a linear classifier on frozen features). Performance is compared against a model trained on the full dataset.

## Expected Results

The distilled datasets should achieve competitive performance compared to training on the full dataset, while using only 0.2-2% of the original data:

- **CIFAR-10/100**: Expect ~60-80% accuracy retention with 100 images
- **Stanford Dogs**: Expect ~50-70% accuracy retention with 240 images (2 per class)

Exact results depend on computational resources and training time.

## Troubleshooting

### CUDA Out of Memory
If you encounter GPU memory errors during Stanford Dogs distillation:
1. Reduce `model_pool.size_L` in the config file (e.g., from 10 to 5)
2. Reduce batch size in evaluation scripts

### Dataset Not Found
Ensure your dataset directory structure matches the format specified in the Setup section exactly.

### Import Errors
Make sure all required packages are installed:
```bash
pip install torch torchvision tqdm pyyaml scikit-learn pillow numpy
```

## Paper Reference

```bibtex
@article{yu2024boost,
  title={Boost Self-Supervised Dataset Distillation via Parameterization, Predefined Augmentation, and Approximation},
  author={Yu, Sheng-Feng and Yao, Jia-Jiun and Chiu, Wei-Chen},
  journal={arXiv preprint arXiv:2507.21455},
  year={2024}
}
```