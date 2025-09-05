import os
import sys
import argparse
import json
import time
import numpy as np
from datetime import datetime
import yaml

from main_evaluate import evaluate_models

def run_multiple_evaluations(config_path, num_runs=3, save_results=True):
    """
    Run multiple evaluation runs to get statistical significance.
    Following the paper's evaluation protocol with multiple seeds.
    """
    print("="*80)
    print("COMPREHENSIVE EVALUATION - MULTIPLE RUNS FOR STATISTICAL SIGNIFICANCE")
    print("="*80)
    
    all_results = []
    
    for run_idx in range(num_runs):
        print(f"\n{'='*20} RUN {run_idx + 1}/{num_runs} {'='*20}")
        
        # Set different random seed for each run
        np.random.seed(42 + run_idx)
        
        try:
            results = evaluate_models(config_path)
            if results:
                all_results.append(results)
                print(f"Run {run_idx + 1} completed successfully")
            else:
                print(f"Run {run_idx + 1} failed")
        except Exception as e:
            print(f"Error in run {run_idx + 1}: {str(e)}")
            continue
    
    if not all_results:
        print("All evaluation runs failed!")
        return None
    
    # Calculate statistics
    full_accuracies = [r['full_dataset']['accuracy'] for r in all_results]
    distilled_accuracies = [r['distilled_dataset']['accuracy'] for r in all_results]
    full_times = [r['full_dataset']['training_time'] for r in all_results]
    distilled_times = [r['distilled_dataset']['training_time'] for r in all_results]
    
    # Statistical analysis
    stats = {
        'num_runs': len(all_results),
        'full_dataset': {
            'accuracy_mean': np.mean(full_accuracies),
            'accuracy_std': np.std(full_accuracies),
            'accuracy_min': np.min(full_accuracies),
            'accuracy_max': np.max(full_accuracies),
            'time_mean': np.mean(full_times),
            'time_std': np.std(full_times)
        },
        'distilled_dataset': {
            'accuracy_mean': np.mean(distilled_accuracies),
            'accuracy_std': np.std(distilled_accuracies),
            'accuracy_min': np.min(distilled_accuracies),
            'accuracy_max': np.max(distilled_accuracies),
            'time_mean': np.mean(distilled_times),
            'time_std': np.std(distilled_times)
        }
    }
    
    # Calculate derived metrics
    accuracy_ratios = [d/f for d, f in zip(distilled_accuracies, full_accuracies)]
    time_ratios = [d/f for d, f in zip(distilled_times, full_times)]
    
    stats['efficiency_metrics'] = {
        'accuracy_retention_mean': np.mean(accuracy_ratios),
        'accuracy_retention_std': np.std(accuracy_ratios),
        'time_ratio_mean': np.mean(time_ratios),
        'time_ratio_std': np.std(time_ratios),
        'speedup_mean': np.mean([1/tr for tr in time_ratios]),
        'speedup_std': np.std([1/tr for tr in time_ratios])
    }
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICAL RESULTS")
    print("="*80)
    
    print(f"\nFull Dataset Training ({stats['num_runs']} runs):")
    print(f"  Accuracy: {stats['full_dataset']['accuracy_mean']:.4f} ± {stats['full_dataset']['accuracy_std']:.4f}")
    print(f"  Range: [{stats['full_dataset']['accuracy_min']:.4f}, {stats['full_dataset']['accuracy_max']:.4f}]")
    print(f"  Training Time: {stats['full_dataset']['time_mean']:.2f} ± {stats['full_dataset']['time_std']:.2f} seconds")
    
    print(f"\nDistilled Dataset Training ({stats['num_runs']} runs):")
    print(f"  Accuracy: {stats['distilled_dataset']['accuracy_mean']:.4f} ± {stats['distilled_dataset']['accuracy_std']:.4f}")
    print(f"  Range: [{stats['distilled_dataset']['accuracy_min']:.4f}, {stats['distilled_dataset']['accuracy_max']:.4f}]")
    print(f"  Training Time: {stats['distilled_dataset']['time_mean']:.2f} ± {stats['distilled_dataset']['time_std']:.2f} seconds")
    
    print(f"\nEfficiency Metrics:")
    print(f"  Accuracy Retention: {stats['efficiency_metrics']['accuracy_retention_mean']:.4f} ± {stats['efficiency_metrics']['accuracy_retention_std']:.4f}")
    print(f"  Training Speedup: {stats['efficiency_metrics']['speedup_mean']:.2f}x ± {stats['efficiency_metrics']['speedup_std']:.2f}x")
    print(f"  Time Ratio: {stats['efficiency_metrics']['time_ratio_mean']:.4f} ± {stats['efficiency_metrics']['time_ratio_std']:.4f}")
    
    # Performance analysis
    mean_retention = stats['efficiency_metrics']['accuracy_retention_mean']
    mean_speedup = stats['efficiency_metrics']['speedup_mean']
    
    print(f"\nPerformance Analysis:")
    if mean_retention > 0.85:
        performance_grade = "EXCELLENT"
    elif mean_retention > 0.70:
        performance_grade = "GOOD"
    elif mean_retention > 0.50:
        performance_grade = "ACCEPTABLE"
    else:
        performance_grade = "POOR"
    
    print(f"  Overall Performance: {performance_grade}")
    print(f"  Retention Rate: {mean_retention*100:.1f}% (±{stats['efficiency_metrics']['accuracy_retention_std']*100:.1f}%)")
    print(f"  Training Efficiency: {mean_speedup:.1f}x speedup")
    
    # Confidence intervals (95%)
    print(f"\n95% Confidence Intervals:")
    acc_ci = 1.96 * stats['distilled_dataset']['accuracy_std'] / np.sqrt(stats['num_runs'])
    ret_ci = 1.96 * stats['efficiency_metrics']['accuracy_retention_std'] / np.sqrt(stats['num_runs'])
    print(f"  Distilled Accuracy: {stats['distilled_dataset']['accuracy_mean']:.4f} ± {acc_ci:.4f}")
    print(f"  Accuracy Retention: {stats['efficiency_metrics']['accuracy_retention_mean']:.4f} ± {ret_ci:.4f}")
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"evaluation_results_{timestamp}.json"
        
        save_data = {
            'timestamp': timestamp,
            'config_path': config_path,
            'statistics': stats,
            'raw_results': all_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION COMPLETED")
    print("="*80)
    
    return stats

def validate_implementation(config_path):
    """
    Validate the implementation against paper specifications.
    """
    print("="*80)
    print("IMPLEMENTATION VALIDATION AGAINST PAPER")
    print("="*80)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    validation_results = {
        'config_validation': True,
        'issues': []
    }
    
    # Check critical hyperparameters
    required_params = {
        'distillation.storage_budget_N': config.get('distillation', {}).get('storage_budget_N'),
        'parametrization.image_bases_U': config.get('parametrization', {}).get('image_bases_U'),
        'parametrization.repr_bases_V': config.get('parametrization', {}).get('repr_bases_V'),
        'models.approximation_mlp.hidden_dim': config.get('models', {}).get('approximation_mlp', {}).get('hidden_dim'),
        'training.linear_eval_lr': config.get('training', {}).get('linear_eval_lr')
    }
    
    print("Configuration Validation:")
    for param, value in required_params.items():
        if value is None:
            validation_results['issues'].append(f"Missing parameter: {param}")
            print(f"  ❌ {param}: MISSING")
            validation_results['config_validation'] = False
        else:
            print(f"  ✅ {param}: {value}")
    
    # Check augmentation strategy
    augs = config.get('augmentations', {})
    print(f"\nAugmentation Strategy:")
    print(f"  ✅ Rotation: {augs.get('rotate', 'Not configured')}")
    print(f"  ✅ Color Jitter: {'Configured' if 'color_jitter' in augs else 'Not configured'}")
    print(f"  ✅ Random Crop: {'Configured' if 'random_crop' in augs else 'Not configured'}")
    
    # Check file existence
    print(f"\nFile Validation:")
    required_files = [
        'main_distill.py',
        'main_evaluate.py', 
        'models.py',
        'utils.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}: EXISTS")
        else:
            print(f"  ❌ {file}: MISSING")
            validation_results['issues'].append(f"Missing file: {file}")
    
    # Check teacher model
    teacher_path = config.get('models', {}).get('teacher', {}).get('path')
    if teacher_path and os.path.exists(teacher_path):
        print(f"  ✅ Teacher model: {teacher_path}")
    else:
        print(f"  ⚠️  Teacher model: {teacher_path} (may need to be trained)")
    
    if validation_results['config_validation'] and not validation_results['issues']:
        print(f"\n✅ IMPLEMENTATION VALIDATION: PASSED")
    else:
        print(f"\n❌ IMPLEMENTATION VALIDATION: FAILED")
        print("Issues found:")
        for issue in validation_results['issues']:
            print(f"  - {issue}")
    
    return validation_results

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Evaluation of Self-Supervised Dataset Distillation')
    parser.add_argument('--config', default='configs/cifar100.yaml', help='Path to config file')
    parser.add_argument('--runs', type=int, default=3, help='Number of evaluation runs')
    parser.add_argument('--validate-only', action='store_true', help='Only validate implementation')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation step')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found!")
        sys.exit(1)
    
    # Validate implementation
    if not args.skip_validation:
        validation_results = validate_implementation(args.config)
        if not validation_results['config_validation']:
            print("Implementation validation failed. Please fix issues before running evaluation.")
            if not args.validate_only:
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    sys.exit(1)
    
    if args.validate_only:
        return
    
    # Run comprehensive evaluation
    print(f"\nStarting comprehensive evaluation with {args.runs} runs...")
    stats = run_multiple_evaluations(args.config, num_runs=args.runs)
    
    if stats:
        print(f"\nEvaluation completed successfully!")
        print(f"Mean accuracy retention: {stats['efficiency_metrics']['accuracy_retention_mean']*100:.1f}%")
        print(f"Mean training speedup: {stats['efficiency_metrics']['speedup_mean']:.1f}x")
    else:
        print("Evaluation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
