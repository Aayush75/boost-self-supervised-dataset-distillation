"""
Comprehensive Evaluation Script for Self-Supervised Dataset Distillation

This script provides a complete evaluation framework comparing ResNet18 models 
trained on the full CIFAR100 dataset versus the distilled dataset.

Two evaluation approaches are implemented:
1. Self-supervised evaluation (feature extraction + linear evaluation)
2. Supervised classification evaluation

Usage:
    python run_evaluation.py --mode [ssl|classification|both]
"""

import os
import argparse
import yaml
import time
import subprocess
import sys

def run_ssl_evaluation():
    """Run self-supervised learning evaluation."""
    print("="*80)
    print("RUNNING SELF-SUPERVISED LEARNING EVALUATION")
    print("="*80)
    
    try:
        import main_evaluate
        results = main_evaluate.evaluate_models('configs/cifar100.yaml')
        return results
    except Exception as e:
        print(f"Error running SSL evaluation: {e}")
        return None

def run_classification_evaluation():
    """Run classification evaluation."""
    print("="*80)
    print("RUNNING CLASSIFICATION EVALUATION")
    print("="*80)
    
    try:
        import pretrain_classification
        results = pretrain_classification.compare_classification_performance('configs/cifar100.yaml')
        return results
    except Exception as e:
        print(f"Error running classification evaluation: {e}")
        return None

def check_prerequisites():
    """Check if distilled dataset exists."""
    config_path = 'configs/cifar100.yaml'
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    asset_dir = config['saving']['distilled_assets_dir']
    
    if not os.path.exists(asset_dir):
        print(f"Error: Distilled assets not found at {asset_dir}")
        print("Please run the following command first to generate distilled dataset:")
        print("python main_distill.py")
        return False
    
    required_files = [
        'distilled_data.pth',
        'approx_net_rot_90.pth',
        'approx_net_rot_180.pth',
        'approx_net_rot_270.pth'
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(asset_dir, file)):
            print(f"Error: Required file {file} not found in {asset_dir}")
            print("Please run main_distill.py to regenerate the distilled dataset.")
            return False
    
    print("✓ All prerequisites satisfied!")
    return True

def print_summary(ssl_results=None, classification_results=None):
    """Print a comprehensive summary of all results."""
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*80)
    
    if ssl_results:
        print("\n📊 SELF-SUPERVISED LEARNING RESULTS:")
        print("-" * 50)
        print(f"Full Dataset Model:")
        print(f"  ├─ Linear Evaluation Accuracy: {ssl_results['full_dataset']['accuracy']:.4f}")
        print(f"  └─ Training Time: {ssl_results['full_dataset']['training_time']:.2f}s")
        
        print(f"\nDistilled Dataset Model:")
        print(f"  ├─ Linear Evaluation Accuracy: {ssl_results['distilled_dataset']['accuracy']:.4f}")
        print(f"  └─ Training Time: {ssl_results['distilled_dataset']['training_time']:.2f}s")
        
        ssl_accuracy_ratio = ssl_results['distilled_dataset']['accuracy'] / ssl_results['full_dataset']['accuracy']
        ssl_time_ratio = ssl_results['distilled_dataset']['training_time'] / ssl_results['full_dataset']['training_time']
        
        print(f"\nSSL Efficiency Metrics:")
        print(f"  ├─ Accuracy Retention: {ssl_accuracy_ratio:.4f} ({ssl_accuracy_ratio*100:.2f}%)")
        print(f"  └─ Training Speed-up: {1/ssl_time_ratio:.2f}x")
    
    if classification_results:
        print("\n📊 CLASSIFICATION RESULTS:")
        print("-" * 50)
        print(f"Full Dataset Model:")
        print(f"  ├─ Best Test Accuracy: {classification_results['full_dataset']['best_accuracy']:.2f}%")
        print(f"  ├─ Final Test Accuracy: {classification_results['full_dataset']['final_accuracy']:.2f}%")
        print(f"  └─ Training Time: {classification_results['full_dataset']['training_time']:.2f}s")
        
        print(f"\nDistilled Dataset Model:")
        print(f"  ├─ Best Test Accuracy: {classification_results['distilled_dataset']['best_accuracy']:.2f}%")
        print(f"  ├─ Final Test Accuracy: {classification_results['distilled_dataset']['final_accuracy']:.2f}%")
        print(f"  └─ Training Time: {classification_results['distilled_dataset']['training_time']:.2f}s")
        
        cls_accuracy_ratio = classification_results['distilled_dataset']['final_accuracy'] / classification_results['full_dataset']['final_accuracy']
        cls_time_ratio = classification_results['distilled_dataset']['training_time'] / classification_results['full_dataset']['training_time']
        
        print(f"\nClassification Efficiency Metrics:")
        print(f"  ├─ Accuracy Retention: {cls_accuracy_ratio:.4f} ({cls_accuracy_ratio*100:.2f}%)")
        print(f"  └─ Training Speed-up: {1/cls_time_ratio:.2f}x")
    
    if ssl_results and classification_results:
        print("\n🔍 COMPARATIVE ANALYSIS:")
        print("-" * 50)
        ssl_retention = ssl_results['distilled_dataset']['accuracy'] / ssl_results['full_dataset']['accuracy']
        cls_retention = classification_results['distilled_dataset']['final_accuracy'] / classification_results['full_dataset']['final_accuracy']
        
        print(f"Method Comparison:")
        print(f"  ├─ SSL Accuracy Retention: {ssl_retention*100:.2f}%")
        print(f"  ├─ Classification Accuracy Retention: {cls_retention*100:.2f}%")
        
        if ssl_retention > cls_retention:
            print(f"  └─ 🏆 SSL approach shows better retention (+{(ssl_retention-cls_retention)*100:.2f}%)")
        else:
            print(f"  └─ 🏆 Classification approach shows better retention (+{(cls_retention-ssl_retention)*100:.2f}%)")
    
    print("\n" + "="*80)
    print("🎉 EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Evaluate distilled dataset performance')
    parser.add_argument('--mode', choices=['ssl', 'classification', 'both'], default='both',
                        help='Evaluation mode: ssl (self-supervised), classification, or both')
    parser.add_argument('--skip-checks', action='store_true',
                        help='Skip prerequisite checks')
    
    args = parser.parse_args()
    
    print("🚀 Starting Comprehensive Evaluation of Self-Supervised Dataset Distillation")
    print("📝 Paper: Boost Self-Supervised Dataset Distillation via Parameterization, Predefined Augmentation, and Approximation")
    print()
    
    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites():
            print("\n❌ Prerequisites not satisfied. Exiting...")
            return
    
    ssl_results = None
    classification_results = None
    
    # Run evaluations based on mode
    if args.mode in ['ssl', 'both']:
        ssl_results = run_ssl_evaluation()
        if ssl_results is None:
            print("❌ SSL evaluation failed!")
    
    if args.mode in ['classification', 'both']:
        classification_results = run_classification_evaluation()
        if classification_results is None:
            print("❌ Classification evaluation failed!")
    
    # Print comprehensive summary
    print_summary(ssl_results, classification_results)
    
    # Save results
    if ssl_results or classification_results:
        results_file = 'evaluation_results.txt'
        with open(results_file, 'w') as f:
            f.write("Evaluation Results Summary\n")
            f.write("="*50 + "\n\n")
            
            if ssl_results:
                f.write("Self-Supervised Learning Results:\n")
                f.write(f"Full Dataset Accuracy: {ssl_results['full_dataset']['accuracy']:.4f}\n")
                f.write(f"Distilled Dataset Accuracy: {ssl_results['distilled_dataset']['accuracy']:.4f}\n")
                f.write(f"SSL Accuracy Retention: {ssl_results['distilled_dataset']['accuracy']/ssl_results['full_dataset']['accuracy']*100:.2f}%\n\n")
            
            if classification_results:
                f.write("Classification Results:\n")
                f.write(f"Full Dataset Accuracy: {classification_results['full_dataset']['final_accuracy']:.2f}%\n")
                f.write(f"Distilled Dataset Accuracy: {classification_results['distilled_dataset']['final_accuracy']:.2f}%\n")
                f.write(f"Classification Accuracy Retention: {classification_results['distilled_dataset']['final_accuracy']/classification_results['full_dataset']['final_accuracy']*100:.2f}%\n")
        
        print(f"\n📄 Results saved to {results_file}")

if __name__ == "__main__":
    main()
