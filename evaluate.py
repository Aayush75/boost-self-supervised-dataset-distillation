#!/usr/bin/env python3
"""
Quick Start Guide for Evaluating Distilled Dataset

This script provides a simple interface to run the evaluation.
"""

import os
import sys

def check_and_run():
    """Check prerequisites and run evaluation."""
    
    print("🔍 Checking prerequisites...")
    
    # Check if distilled assets exist
    asset_dir = "./distilled_assets/cifar100_N100"
    if not os.path.exists(asset_dir):
        print("❌ Distilled assets not found!")
        print("📝 Please run the following command first:")
        print("   python main_distill.py")
        return False
    
    # Check required files
    required_files = [
        "distilled_data.pth",
        "approx_net_rot_90.pth", 
        "approx_net_rot_180.pth",
        "approx_net_rot_270.pth"
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(asset_dir, file)):
            print(f"❌ Missing required file: {file}")
            print("📝 Please run: python main_distill.py")
            return False
    
    # Check teacher model
    teacher_path = "./teacher_models/resnet18_barlow_twins_cifar100.pth"
    if not os.path.exists(teacher_path):
        print("❌ Teacher model not found!")
        print("📝 Please run the following command first:")
        print("   python pretrain.py")
        return False
    
    print("✅ All prerequisites satisfied!")
    return True

def main():
    print("🚀 ResNet18 Full vs Distilled Dataset Evaluation")
    print("=" * 60)
    
    if not check_and_run():
        print("\n❌ Prerequisites not met. Please follow the setup instructions.")
        return
    
    print("\n🔄 Starting evaluation...")
    print("\nThis will compare ResNet18 models trained on:")
    print("1️⃣  Full CIFAR100 dataset (50,000 images)")
    print("2️⃣  Distilled dataset (100 images)")
    print("\nBoth self-supervised and classification approaches will be tested.")
    
    # Ask user for confirmation
    response = input("\n❓ Continue with evaluation? (y/n): ").lower().strip()
    
    if response != 'y' and response != 'yes':
        print("Evaluation cancelled.")
        return
    
    # Import and run evaluation
    try:
        print("\n📊 Running comprehensive evaluation...")
        import run_evaluation
        # Simulate command line args
        sys.argv = ['run_evaluation.py', '--mode', 'both']
        run_evaluation.main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required packages are installed.")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main()
