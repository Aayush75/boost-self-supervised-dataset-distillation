#!/usr/bin/env python3
"""
Test script to verify Stanford Dogs dataset loading
"""

from utils import get_dataset
import torch

def test_stanford_dogs():
    print("Testing Stanford Dogs Dataset Loading...")
    print("="*60)
    
    try:
        # Test dataset loading
        train_dataset, test_dataset = get_dataset('STANFORD_DOGS', data_dir='.')
        
        print(f" Successfully loaded Stanford Dogs dataset!")
        print(f" Training samples: {len(train_dataset)}")
        print(f" Testing samples: {len(test_dataset)}")
        print(f" Number of classes: {len(train_dataset.classes)}")
        
        # Test loading a few samples
        print("\n Testing sample loading...")
        for i in range(min(3, len(train_dataset))):
            image, label = train_dataset[i]
            print(f"   Sample {i+1}: Image shape: {image.shape}, Label: {label}")
            
        print("\n Sample class names:")
        for i in range(min(5, len(train_dataset.classes))):
            print(f"   Class {i}: {train_dataset.classes[i]}")
            
        print("\n Stanford Dogs dataset test completed successfully!")
        return True
        
    except Exception as e:
        print(f" Error testing Stanford Dogs dataset: {e}")
        return False

if __name__ == "__main__":
    success = test_stanford_dogs()
    if success:
        print("\n You can now use Stanford Dogs dataset for training and distillation!")
        print("   Run: python pretrain.py stanford_dogs")
        print("   Then: python main_distill.py --config configs/stanford_dogs.yaml")
    else:
        print("\n Please check the dataset path and structure.")