#!/usr/bin/env python3
"""
Calculate Token Statistics for Benchmark (Accurate Version)

Features:
1. Calculate text and image tokens for each sample using real tokenizer
2. Extract actual model outputs from VLMEvalKit and calculate real output tokens
3. Generate complete statistical reports and CSV files
4. Calculate actual costs based on tokens (input_tokens × input_price + output_tokens × output_price)

Usage:
    python tools/compute_token_stats.py \
        --dataset_dir . \
        --vlmevalkit_dir /path/to/VLMEvalKit/outputs \
        --output_dir reports/token_statistics
"""

import argparse
import json
import pickle
import sys
import yaml
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from routers.utils.token_stats import (
    aggregate_token_stats,
)

# Use the batch-optimized version
from routers.utils.token_stats_batch import (
    compute_dataset_token_stats_batch as compute_dataset_token_stats,
    compute_token_based_costs_batch as compute_token_based_costs
)


def load_model_list(dataset_dir: Path) -> List[str]:
    """Load model list from config file (preferred) or model_index.pkl (fallback)"""
    # Prefer loading from config (supports dynamic extension)
    config_file = dataset_dir / "config" / "models.yaml"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            models = list(config['models'].keys())
            print(f"✓ Loaded {len(models)} models from config: {config_file}")
            return models
        except Exception as e:
            print(f"⚠️  Failed to load models from config: {e}")
    
    # Fallback: load from model_index.pkl
    model_index_file = dataset_dir / "data/registry/model_index.pkl"
    if model_index_file.exists():
        with open(model_index_file, 'rb') as f:
            models = pickle.load(f)
        print(f"✓ Loaded {len(models)} models from model_index.pkl (fallback)")
        return models
    else:
        print("❌ Error: no model list file found")
        print(f"   - Config file: {config_file}")
        print(f"   - Index file: {model_index_file}")
        return []


def load_pricing_config(pricing_file: Path) -> Dict:
    """Load pricing config from YAML file"""
    if not pricing_file.exists():
        print(f"⚠️  Pricing file not found: {pricing_file}, using default values")
        return None
    
    with open(pricing_file, 'r') as f:
        config = yaml.safe_load(f)
    
    pricing = config.get('pricing', {})
    pricing_dict = {}
    
    for model, prices in pricing.items():
        if prices.get('type') == 'token_based':
            pricing_dict[model] = {
                'input': prices.get('input_price_per_1m', 1.0),
                'output': prices.get('output_price_per_1m', 1.0)
            }
    
    return pricing_dict if pricing_dict else None


def generate_summary_report(
    input_tokens_df: pd.DataFrame,
    output_tokens_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    output_file: Path
):
    """Generate summary report (in English)"""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("Token Statistics Summary Report")
    report_lines.append("="*80)
    report_lines.append("")
    
    # 1. Overall statistics
    report_lines.append("## 1. Overall Statistics")
    report_lines.append("")
    total_samples = len(input_tokens_df)
    total_text_tokens = input_tokens_df['text_tokens'].sum()
    total_image_tokens = input_tokens_df['image_tokens'].sum()
    total_input_tokens = input_tokens_df['total_input_tokens'].sum()
    
    # Calculate total output tokens across all models
    total_output_tokens = 0
    num_models = 0
    if len(output_tokens_df) > 0:
        total_output_tokens = output_tokens_df['output_tokens'].sum()
        num_models = output_tokens_df['model'].nunique()
    
    # Grand total: input + output
    grand_total_tokens = total_input_tokens + total_output_tokens
    
    report_lines.append(f"  Total Samples: {total_samples:,}")
    report_lines.append(f"  Total Models: {num_models}")
    report_lines.append("")
    report_lines.append(f"  Total Text Tokens (Input): {total_text_tokens:,}")
    report_lines.append(f"  Total Image Tokens (Input): {total_image_tokens:,}")
    report_lines.append(f"  Total Input Tokens: {total_input_tokens:,}")
    report_lines.append(f"  Total Output Tokens (All Models): {total_output_tokens:,}")
    report_lines.append(f"  **GRAND TOTAL (Input + Output): {grand_total_tokens:,}**")
    report_lines.append("")
    report_lines.append(f"  Avg Input Tokens per Sample: {total_input_tokens/total_samples:.1f}")
    if num_models > 0:
        avg_output_per_model_sample = total_output_tokens / (total_samples * num_models) if (total_samples * num_models) > 0 else 0
        report_lines.append(f"  Avg Output Tokens per Model-Sample: {avg_output_per_model_sample:.1f}")
    report_lines.append("")
    
    # 2. Statistics by dataset
    report_lines.append("## 2. Statistics by Dataset")
    report_lines.append("")
    dataset_stats = aggregate_token_stats(input_tokens_df, group_by='dataset')
    dataset_stats = dataset_stats.sort_values('total_input_tokens_sum', ascending=False)
    
    report_lines.append("| Dataset | Samples | Avg Text | Avg Image | Avg Total | Total Tokens |")
    report_lines.append("|---------|---------|----------|-----------|-----------|--------------|")
    
    for _, row in dataset_stats.iterrows():
        report_lines.append(
            f"| {row['dataset']:<20} | {int(row['num_samples']):>7,} | "
            f"{row['text_tokens_mean']:>8.1f} | {row['image_tokens_mean']:>9.1f} | "
            f"{row['total_input_tokens_mean']:>9.1f} | {int(row['total_input_tokens_sum']):>12,} |"
        )
    
    report_lines.append("")
    
    # 3. Statistics by task type
    report_lines.append("## 3. Statistics by Task Type")
    report_lines.append("")
    task_stats = aggregate_token_stats(input_tokens_df, group_by='task_type')
    task_stats = task_stats.sort_values('total_input_tokens_mean', ascending=False)
    
    report_lines.append("| Task Type | Samples | Avg Text | Avg Image | Avg Total |")
    report_lines.append("|-----------|---------|----------|-----------|-----------|")
    
    for _, row in task_stats.iterrows():
        report_lines.append(
            f"| {row['task_type']:<15} | {int(row['num_samples']):>7,} | "
            f"{row['text_tokens_mean']:>8.1f} | {row['image_tokens_mean']:>9.1f} | "
            f"{row['total_input_tokens_mean']:>9.1f} |"
        )
    
    report_lines.append("")
    
    # 4. Model output token statistics
    if len(output_tokens_df) > 0:
        report_lines.append("## 4. Model Output Token Statistics")
        report_lines.append("")
        
        model_output_stats = output_tokens_df.groupby('model').agg({
            'output_tokens': ['mean', 'sum'],
            'is_actual': 'mean'
        }).round(1)
        model_output_stats.columns = ['avg_output_tokens', 'total_output_tokens', 'actual_ratio']
        model_output_stats = model_output_stats.reset_index()
        model_output_stats = model_output_stats.sort_values('avg_output_tokens', ascending=False)
        
        report_lines.append("| Model | Avg Output | Total Output | Actual % |")
        report_lines.append("|-------|------------|--------------|----------|")
        
        for _, row in model_output_stats.iterrows():
            report_lines.append(
                f"| {row['model']:<35} | {row['avg_output_tokens']:>10.1f} | "
                f"{int(row['total_output_tokens']):>12,} | {100*row['actual_ratio']:>7.1f}% |"
            )
        
        report_lines.append("")
    
    # 5. Token-based cost statistics
    if len(cost_df) > 0:
        report_lines.append("## 5. Token-Based Cost Statistics")
        report_lines.append("")
        
        model_cost_stats = cost_df.groupby('model').agg({
            'input_cost': 'sum',
            'output_cost': 'sum',
            'total_cost': ['mean', 'sum']
        }).round(6)
        model_cost_stats.columns = ['total_input_cost', 'total_output_cost', 
                                    'avg_cost_per_sample', 'total_cost']
        model_cost_stats = model_cost_stats.reset_index()
        model_cost_stats = model_cost_stats.sort_values('total_cost', ascending=False)
        
        report_lines.append("| Model | Input Cost($) | Output Cost($) | Avg Cost/Sample($) | Total Cost($) |")
        report_lines.append("|-------|---------------|----------------|--------------------|--------------" + "|")
        
        for _, row in model_cost_stats.iterrows():
            report_lines.append(
                f"| {row['model']:<35} | {row['total_input_cost']:>13.6f} | "
                f"{row['total_output_cost']:>14.6f} | {row['avg_cost_per_sample']:>18.6f} | "
                f"{row['total_cost']:>13.6f} |"
            )
        
        report_lines.append("")
        report_lines.append(f"  **Total Cost**: ${model_cost_stats['total_cost'].sum():.6f}")
        report_lines.append("")
    
    # 6. Token distribution statistics
    report_lines.append("## 6. Token Distribution Statistics")
    report_lines.append("")
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    text_percentiles = np.percentile(input_tokens_df['text_tokens'], percentiles)
    image_percentiles = np.percentile(input_tokens_df['image_tokens'], percentiles)
    total_percentiles = np.percentile(input_tokens_df['total_input_tokens'], percentiles)
    
    report_lines.append("| Percentile | Text Token | Image Token | Total Input Token |")
    report_lines.append("|------------|------------|-------------|-------------------|")
    
    for i, p in enumerate(percentiles):
        report_lines.append(
            f"| P{p:<2} | {text_percentiles[i]:>10.0f} | "
            f"{image_percentiles[i]:>11.0f} | {total_percentiles[i]:>17.0f} |"
        )
    
    report_lines.append("")
    report_lines.append("="*80)
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Also print to console
    print('\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Token Statistics for Benchmark (Accurate Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate token statistics for all datasets
  python tools/compute_token_stats.py \
      --dataset_dir . \
      --vlmevalkit_dir /path/to/VLMEvalKit/outputs
  
  # Specify output directory
  python tools/compute_token_stats.py \
      --dataset_dir . \
      --vlmevalkit_dir /path/to/VLMEvalKit/outputs \
      --output_dir reports/token_statistics
        """
    )
    parser.add_argument('--dataset_dir', default='.',
                       help="Dataset root directory (default: .)")
    parser.add_argument('--vlmevalkit_dir', required=True,
                       help="VLMEvalKit output directory (for extracting actual model predictions)")
    parser.add_argument('--output_dir', default='reports/token_statistics',
                       help="Output directory (default: reports/token_statistics)")
    parser.add_argument('--datasets', nargs='+',
                       help="List of datasets to process (default: all)")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    vlmevalkit_dir = Path(args.vlmevalkit_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("📊 Calculate Token Statistics (Accurate Version)")
    print("="*80)
    print(f"  Dataset dir: {dataset_dir}")
    print(f"  VLMEvalKit dir: {vlmevalkit_dir}")
    print(f"  Output dir: {output_dir}")
    print()
    
    # 1. Calculate input token statistics
    print("📥 Step 1: Calculate Input Token Statistics...")
    benchmark_dir = dataset_dir / "BENCHMARKS"
    
    if not benchmark_dir.exists():
        print(f"❌ Error: BENCHMARKS directory does not exist: {benchmark_dir}")
        return
    
    if not vlmevalkit_dir.exists():
        print(f"⚠️  Warning: VLMEvalKit directory does not exist: {vlmevalkit_dir}")
        print(f"   Will use estimation for output tokens")
    
    # Load model list
    models = load_model_list(dataset_dir)
    if len(models) == 0:
        print("❌ Error: No models found")
        return
    
    print(f"  Models: {', '.join(models)}")
    
    # Calculate token statistics (both input and output)
    input_tokens_df, output_tokens_df = compute_dataset_token_stats(
        benchmark_dir,
        vlmevalkit_dir if vlmevalkit_dir.exists() else None,
        models,
        datasets=args.datasets
    )
    
    if len(input_tokens_df) == 0:
        print("❌ Error: No samples found")
        return
    
    print(f"  ✓ Processed {len(input_tokens_df)} samples")
    
    # Save detailed statistics
    input_stats_file = output_dir / "token_stats_per_sample.csv"
    input_tokens_df.to_csv(input_stats_file, index=False)
    print(f"  ✓ Input token statistics saved: {input_stats_file}")
    
    # Save output token statistics
    if len(output_tokens_df) > 0:
        output_stats_file = output_dir / "output_tokens_per_model.csv"
        output_tokens_df.to_csv(output_stats_file, index=False)
        print(f"  ✓ Output token statistics saved: {output_stats_file}")
        
        # 2. Calculate token-based costs
        print("\n💰 Step 2: Calculate Token-Based Costs...")
        # Load pricing configuration
        pricing_file = Path(args.dataset_dir) / "config" / "pricing.yaml"
        pricing_config = load_pricing_config(pricing_file)
        if pricing_config:
            print(f"  ✓ Loaded pricing from: {pricing_file}")
        else:
            print(f"  ⚠️  Using default pricing values")
        
        cost_df = compute_token_based_costs(input_tokens_df, output_tokens_df, pricing_config)
        
        # Save cost statistics
        cost_file = output_dir / "token_based_costs.csv"
        cost_df.to_csv(cost_file, index=False)
        print(f"  ✓ Token-based costs saved: {cost_file}")
    else:
        cost_df = pd.DataFrame()
        print("  ⚠️  No output tokens, skipping cost calculation")
    
    # 3. Generate aggregated statistics
    print("\n📊 Step 3: Generate Aggregated Statistics...")
    
    # Aggregate by dataset
    dataset_agg = aggregate_token_stats(input_tokens_df, group_by='dataset')
    dataset_agg_file = output_dir / "token_stats_by_dataset.csv"
    dataset_agg.to_csv(dataset_agg_file, index=False)
    print(f"  ✓ By dataset: {dataset_agg_file}")
    
    # Aggregate by task type
    task_agg = aggregate_token_stats(input_tokens_df, group_by='task_type')
    task_agg_file = output_dir / "token_stats_by_task.csv"
    task_agg.to_csv(task_agg_file, index=False)
    print(f"  ✓ By task type: {task_agg_file}")
    
    # 4. Generate summary report
    print("\n📄 Step 4: Generate Summary Report...")
    report_file = output_dir / "token_statistics_report.txt"
    generate_summary_report(input_tokens_df, output_tokens_df, cost_df, report_file)
    print(f"\n  ✓ Summary report saved: {report_file}")
    
    print("\n" + "="*80)
    print("✅ Token Statistics Complete!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print(f"  - Input tokens: {input_stats_file.name}")
    if len(output_tokens_df) > 0:
        print(f"  - Output tokens: {output_stats_file.name}")
        print(f"  - Token costs: {cost_file.name}")
    print(f"  - By dataset: {dataset_agg_file.name}")
    print(f"  - By task: {task_agg_file.name}")
    print(f"  - Report: {report_file.name}")


if __name__ == '__main__':
    main()
