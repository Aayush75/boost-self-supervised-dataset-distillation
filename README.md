# Self-Supervised Dataset Distillation

Implementation of "Boost Self-Supervised Dataset Distillation via Parameterization, Predefined Augmentation, and Approximation" (arXiv:2507.21455v2).

## Overview

This method distills large datasets into compact, synthetic datasets while preserving performance through:
- Parameterization of images and representations via low-dimensional bases
- Predefined augmentations to avoid gradient bias
- Approximation networks for compact representation

## Supported Datasets

- **CIFAR-100**: 32x32, 100 classes
- **Stanford Dogs**: 64x64, 120 dog breeds

## Setup

### Requirements
```bash
pip install torch torchvision tqdm pyyaml scikit-learn pillow numpy
```

### Stanford Dogs Dataset
Download from Kaggle and place in:
```
full_datasets/stanford-dogs/
├── images/Images/
│   ├── n02085620-Chihuahua/
│   ├── n02085782-Japanese_spaniel/
│   └── ... (120 breed folders)
└── annotations/Annotation/
```

## Usage

### 1. Test Dataset Loading (Stanford Dogs only)
```bash
python test_stanford_dogs.py
```

### 2. Train Teacher Model
```bash
# CIFAR-100
python pretrain.py

# Stanford Dogs
python pretrain.py stanford_dogs
```

### 3. Dataset Distillation
```bash
# CIFAR-100
python main_distill.py

# Stanford Dogs
python main_distill.py --config configs/stanford_dogs.yaml
```

### 4. Evaluation
```bash
# CIFAR-100
python main_evaluate.py

# Stanford Dogs
python main_evaluate.py configs/stanford_dogs.yaml
```

## Configuration

### CIFAR-100 (`configs/cifar100.yaml`)
- Storage budget: 100 images
- Resolution: 32x32
- Image bases: 200
- Representation bases: 200

### Stanford Dogs (`configs/stanford_dogs.yaml`)
- Storage budget: 240 images (2 per class)
- Resolution: 64x64
- Image bases: 400
- Representation bases: 400

## Output Files

- `teacher_models/`: Pre-trained teacher models
- `distilled_assets/`: Distilled datasets and approximation networks
- `evaluation_models/`: Evaluation results

## File Structure

```
.
├── configs/              # Configuration files
├── teacher_models/       # Pre-trained teacher models
├── distilled_assets/     # Distilled datasets
├── main_distill.py      # Main distillation script
├── pretrain.py          # Teacher model training
├── main_evaluate.py     # Evaluation script
├── utils.py             # Dataset utilities
└── models.py            # Model definitions
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