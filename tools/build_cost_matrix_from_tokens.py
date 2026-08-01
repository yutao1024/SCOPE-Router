#!/usr/bin/env python3
"""
Build Cost Matrix from Token Statistics and Token-Based Pricing

This script:
1. Loads token-based pricing from config/pricing.yaml
2. Loads token statistics from token_stats reports
3. Calculates per-sample costs for each model
4. Generates cost matrix C for the benchmark
5. Outputs validation information at each step
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm


def load_pricing_config(pricing_file: Path) -> dict:
    """Load token-based pricing from YAML config"""
    print("=" * 80)
    print("📥 Step 1: Load Token-Based Pricing")
    print("=" * 80)
    
    with open(pricing_file, 'r') as f:
        config = yaml.safe_load(f)
    
    pricing = config.get('pricing', {})
    
    print(f"\n✓ Loaded pricing for {len(pricing)} models:")
    for model, prices in pricing.items():
        if prices.get('type') == 'token_based':
            input_price = prices.get('input_price_per_1m', 0)
            output_price = prices.get('output_price_per_1m', 0)
            print(f"  • {model:35s}: ${input_price:.2f}/1M input, ${output_price:.2f}/1M output")
        else:
            print(f"  • {model:35s}: [Not token-based]")
    
    return pricing


def load_token_statistics(token_stats_dir: Path) -> tuple:
    """Load token statistics from reports"""
    print("\n" + "=" * 80)
    print("📥 Step 2: Load Token Statistics")
    print("=" * 80)
    
    # Load input token stats
    input_stats_file = token_stats_dir / "token_stats_per_sample.csv"
    print(f"\n📄 Loading input tokens: {input_stats_file}")
    input_df = pd.read_csv(input_stats_file)
    print(f"  ✓ Loaded {len(input_df)} samples")
    print(f"  ✓ Columns: {input_df.columns.tolist()}")
    
    # Load output token stats
    output_stats_file = token_stats_dir / "output_tokens_per_model.csv"
    print(f"\n📄 Loading output tokens: {output_stats_file}")
    output_df = pd.read_csv(output_stats_file)
    print(f"  ✓ Loaded {len(output_df)} records")
    print(f"  ✓ Columns: {output_df.columns.tolist()}")
    
    # Show sample data
    print("\n📊 Sample Input Token Data (first 3 rows):")
    print(input_df[['sample_id', 'text_tokens', 'image_tokens', 'total_input_tokens']].head(3).to_string(index=False))
    
    print("\n📊 Sample Output Token Data (first 5 rows):")
    print(output_df[['sample_id', 'model', 'output_tokens', 'is_actual']].head(5).to_string(index=False))
    
    # Statistics
    print("\n📈 Token Statistics Summary:")
    print(f"  • Total samples: {len(input_df)}")
    print(f"  • Total model-sample pairs: {len(output_df)}")
    print(f"  • Models: {output_df['model'].nunique()}")
    print(f"  • Avg input tokens/sample: {input_df['total_input_tokens'].mean():.1f}")
    print(f"  • Avg output tokens/sample: {output_df['output_tokens'].mean():.1f}")
    print(f"  • Actual output data: {(output_df['is_actual'].sum() / len(output_df) * 100):.1f}%")
    
    return input_df, output_df


def calculate_sample_costs(input_df: pd.DataFrame, output_df: pd.DataFrame, 
                          pricing: dict) -> pd.DataFrame:
    """Calculate per-sample costs for each model (batch-optimized version)"""
    print("\n" + "=" * 80)
    print("💰 Step 3: Calculate Sample-Level Costs (Batch Processing)")
    print("=" * 80)
    
    # Merge input and output data using vectorized operations
    print("\n🔗 Merging input and output token data...")
    print(f"  Processing {len(output_df):,} output records with {len(input_df):,} input records...")
    
    with tqdm(total=3, desc="Merging data", unit="step") as pbar:
        # Step 1: Merge
        cost_df = output_df.merge(
            input_df[['sample_id', 'total_input_tokens']], 
            on='sample_id', 
            how='left'
        )
        pbar.update(1)
        
        # Step 2: Check for missing input tokens
        missing_input = cost_df['total_input_tokens'].isna().sum()
        if missing_input > 0:
            print(f"  ⚠️  Warning: {missing_input} records have missing input tokens")
            cost_df = cost_df.dropna(subset=['total_input_tokens'])
        pbar.update(1)
        
        # Step 3: Rename for clarity
        cost_df = cost_df.rename(columns={'total_input_tokens': 'input_tokens'})
        pbar.update(1)
    
    print(f"  ✓ Merged {len(cost_df):,} records")
    
    # Add pricing columns in batch (vectorized)
    print("\n💵 Applying pricing information...")
    
    # Create pricing lookup DataFrames for vectorized operations
    valid_models = [m for m in pricing.keys() if pricing[m].get('type') == 'token_based']
    print(f"  Found {len(valid_models)} models with token-based pricing")
    
    pricing_data = []
    for model in tqdm(valid_models, desc="Building pricing table", unit="model"):
        model_pricing = pricing[model]
        pricing_data.append({
            'model': model,
            'input_price_per_1m': model_pricing.get('input_price_per_1m', 0),
            'output_price_per_1m': model_pricing.get('output_price_per_1m', 0)
        })
    
    pricing_df = pd.DataFrame(pricing_data)
    
    # Filter out models without token-based pricing
    models_without_pricing = set(cost_df['model'].unique()) - set(valid_models)
    if models_without_pricing:
        print(f"  ⚠️  Warning: {len(models_without_pricing)} models without token-based pricing: {models_without_pricing}")
        cost_df = cost_df[cost_df['model'].isin(valid_models)]
    
    # Merge pricing info
    print(f"  Merging pricing data for {len(cost_df):,} records...")
    cost_df = cost_df.merge(pricing_df, on='model', how='left')
    print(f"  ✓ Applied pricing to {len(cost_df):,} records")
    
    # Vectorized cost calculation (no loops!)
    print("\n🧮 Calculating costs (vectorized)...")
    unique_models = cost_df['model'].nunique()
    unique_samples = cost_df['sample_id'].nunique()
    print(f"  Processing {unique_samples:,} samples × {unique_models} models = {len(cost_df):,} calculations")
    
    with tqdm(total=3, desc="Computing costs", unit="step") as pbar:
        cost_df['input_cost_usd'] = (cost_df['input_tokens'] / 1_000_000) * cost_df['input_price_per_1m']
        pbar.update(1)
        
        cost_df['output_cost_usd'] = (cost_df['output_tokens'] / 1_000_000) * cost_df['output_price_per_1m']
        pbar.update(1)
        
        cost_df['total_cost_usd'] = cost_df['input_cost_usd'] + cost_df['output_cost_usd']
        pbar.update(1)
    
    # Drop temporary pricing columns
    cost_df = cost_df.drop(columns=['input_price_per_1m', 'output_price_per_1m'])
    print(f"  ✓ Completed {len(cost_df):,} cost calculations")
    
    print(f"\n✓ Calculated costs for {len(cost_df)} sample-model pairs")
    
    # Show sample costs
    print("\n📊 Sample Cost Data (first 10 rows):")
    display_df = cost_df[['sample_id', 'model', 'input_tokens', 'output_tokens', 
                          'input_cost_usd', 'output_cost_usd', 'total_cost_usd']].head(10)
    print(display_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    
    # Cost statistics by model
    print("\n📈 Cost Statistics by Model:")
    print("-" * 120)
    print(f"{'Model':<35s} {'Avg Input':<12s} {'Avg Output':<12s} {'Avg Total':<12s} {'Min Cost':<12s} {'Max Cost':<12s}")
    print("-" * 120)
    
    for model in sorted(cost_df['model'].unique()):
        model_df = cost_df[cost_df['model'] == model]
        avg_input = model_df['input_cost_usd'].mean()
        avg_output = model_df['output_cost_usd'].mean()
        avg_total = model_df['total_cost_usd'].mean()
        min_cost = model_df['total_cost_usd'].min()
        max_cost = model_df['total_cost_usd'].max()
        
        print(f"{model:<35s} ${avg_input:>10.6f}  ${avg_output:>10.6f}  ${avg_total:>10.6f}  ${min_cost:>10.6f}  ${max_cost:>10.6f}")
    
    print("-" * 120)
    
    # Overall statistics
    print(f"\n📊 Overall Cost Statistics:")
    print(f"  • Total cost (all samples, all models): ${cost_df['total_cost_usd'].sum():.2f}")
    print(f"  • Average cost per sample-model pair: ${cost_df['total_cost_usd'].mean():.6f}")
    print(f"  • Cost range: ${cost_df['total_cost_usd'].min():.6f} - ${cost_df['total_cost_usd'].max():.6f}")
    
    return cost_df


def load_valid_samples(dataset_dir: Path) -> list:
    """Load valid sample IDs from meta.parquet"""
    meta_file = dataset_dir / "data/registry/meta.parquet"
    if not meta_file.exists():
        print(f"  ⚠️  Warning: meta.parquet not found at {meta_file}, using all samples")
        return None
    
    meta = pd.read_parquet(meta_file)
    valid_samples = meta['sample_id'].unique().tolist()
    print(f"  ✓ Loaded {len(valid_samples)} valid samples from meta.parquet")
    return valid_samples


def build_cost_matrix(cost_df: pd.DataFrame, input_df: pd.DataFrame, 
                     valid_samples: list = None) -> tuple:
    """Build cost matrix C (samples x models)"""
    print("\n" + "=" * 80)
    print("🏗️  Step 4: Build Cost Matrix C")
    print("=" * 80)
    
    # Filter to valid samples if provided
    if valid_samples is not None:
        print(f"\n🔍 Filtering to valid samples...")
        print(f"  • Total samples in data: {input_df['sample_id'].nunique()}")
        print(f"  • Valid samples from meta: {len(valid_samples)}")
        
        # Filter input_df and cost_df
        input_df = input_df[input_df['sample_id'].isin(valid_samples)].copy()
        cost_df = cost_df[cost_df['sample_id'].isin(valid_samples)].copy()
        
        print(f"  ✓ Filtered to {len(input_df)} samples")
        print(f"  ✓ Cost records: {len(cost_df)}")
    
    # Get unique samples and models
    samples = sorted(input_df['sample_id'].unique())
    models = sorted(cost_df['model'].unique())
    
    print(f"\n📐 Matrix dimensions:")
    print(f"  • Samples: {len(samples)}")
    print(f"  • Models: {len(models)}")
    print(f"  • Matrix shape: ({len(samples)}, {len(models)})")
    
    # Initialize cost matrix
    C = np.zeros((len(samples), len(models)), dtype=np.float32)
    
    # Fill matrix
    sample_to_idx = {s: i for i, s in enumerate(samples)}
    model_to_idx = {m: i for i, m in enumerate(models)}
    
    print(f"\n🔧 Filling cost matrix...")
    missing_count = 0
    
    for _, row in cost_df.iterrows():
        sample_id = row['sample_id']
        model = row['model']
        cost = row['total_cost_usd']
        
        if sample_id in sample_to_idx and model in model_to_idx:
            i = sample_to_idx[sample_id]
            j = model_to_idx[model]
            C[i, j] = cost
        else:
            missing_count += 1
    
    if missing_count > 0:
        print(f"  ⚠️  Warning: {missing_count} entries could not be mapped to matrix")
    
    print(f"  ✓ Cost matrix filled")
    
    # Validate matrix
    print(f"\n✅ Cost Matrix Validation:")
    print(f"  • Non-zero entries: {np.count_nonzero(C)} / {C.size} ({np.count_nonzero(C)/C.size*100:.1f}%)")
    print(f"  • Zero entries: {np.sum(C == 0)} ({np.sum(C == 0)/C.size*100:.1f}%)")
    print(f"  • Min cost: ${C[C > 0].min():.6f}")
    print(f"  • Max cost: ${C.max():.6f}")
    print(f"  • Mean cost: ${C[C > 0].mean():.6f}")
    print(f"  • Median cost: ${np.median(C[C > 0]):.6f}")
    
    # Show sample of matrix
    print(f"\n📊 Cost Matrix Sample (first 5 samples, first 5 models):")
    sample_models = models[:5]
    print(f"\n{'Sample ID':<40s} " + "  ".join([f"{m[:15]:>15s}" for m in sample_models]))
    print("-" * 140)
    for i in range(min(5, len(samples))):
        sample_id = samples[i]
        costs_str = "  ".join([f"${C[i, j]:>14.6f}" for j in range(min(5, len(models)))])
        print(f"{sample_id[:40]:<40s} {costs_str}")
    
    # Return filtered cost_df as well
    return C, samples, models, cost_df


def save_results(C: np.ndarray, samples: list, models: list, 
                cost_df: pd.DataFrame, output_dir: Path):
    """Save cost matrix and related data"""
    print("\n" + "=" * 80)
    print("💾 Step 5: Save Results")
    print("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate cost bounds (cmin, cmax) for Arena Score
    # Use per-model average costs
    per_model_avg_costs = []
    for j, model in enumerate(models):
        model_costs = C[:, j]
        avg_cost = model_costs[model_costs > 0].mean()
        per_model_avg_costs.append(avg_cost)
    
    cmin = float(np.min(per_model_avg_costs))
    cmax = float(np.max(per_model_avg_costs))
    
    # Save cost bounds config for Arena Score
    cost_bounds_file = output_dir / "cost_bounds.json"
    import json
    cost_bounds = {
        'cmin': cmin,
        'cmax': cmax,
        'description': 'Cost bounds for Arena Score calculation',
        'cmin_model': models[np.argmin(per_model_avg_costs)],
        'cmax_model': models[np.argmax(per_model_avg_costs)],
        'per_model_avg_costs': {
            model: float(cost) 
            for model, cost in zip(models, per_model_avg_costs)
        }
    }
    with open(cost_bounds_file, 'w') as f:
        json.dump(cost_bounds, f, indent=2)
    print(f"\n✓ Cost bounds saved: {cost_bounds_file}")
    print(f"  cmin: ${cmin:.6f} ({cost_bounds['cmin_model']})")
    print(f"  cmax: ${cmax:.6f} ({cost_bounds['cmax_model']})")
    
    # Save cost matrix
    cost_matrix_file = output_dir / "C.npy"
    np.save(cost_matrix_file, C)
    print(f"\n✓ Cost matrix saved: {cost_matrix_file}")
    print(f"  Shape: {C.shape}")
    print(f"  Dtype: {C.dtype}")
    
    # Save sample IDs
    samples_file = output_dir / "sample_ids.pkl"
    with open(samples_file, 'wb') as f:
        pickle.dump(samples, f)
    print(f"\n✓ Sample IDs saved: {samples_file}")
    print(f"  Count: {len(samples)}")
    
    # Save model names
    models_file = output_dir / "model_names.pkl"
    with open(models_file, 'wb') as f:
        pickle.dump(models, f)
    print(f"\n✓ Model names saved: {models_file}")
    print(f"  Count: {len(models)}")
    print(f"  Models: {', '.join(models)}")
    
    # Save detailed cost dataframe
    cost_detail_file = output_dir / "cost_details.csv"
    cost_df.to_csv(cost_detail_file, index=False, float_format='%.8f')
    print(f"\n✓ Cost details saved: {cost_detail_file}")
    print(f"  Records: {len(cost_df)}")
    
    # Generate summary report
    summary_file = output_dir / "cost_matrix_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Cost Matrix Summary Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated from token-based pricing\n\n")
        
        f.write(f"Matrix Dimensions:\n")
        f.write(f"  • Samples: {len(samples)}\n")
        f.write(f"  • Models: {len(models)}\n")
        f.write(f"  • Matrix shape: {C.shape}\n\n")
        
        f.write(f"Cost Statistics:\n")
        f.write(f"  • Total entries: {C.size}\n")
        f.write(f"  • Non-zero entries: {np.count_nonzero(C)} ({np.count_nonzero(C)/C.size*100:.1f}%)\n")
        f.write(f"  • Min cost: ${C[C > 0].min():.6f}\n")
        f.write(f"  • Max cost: ${C.max():.6f}\n")
        f.write(f"  • Mean cost: ${C[C > 0].mean():.6f}\n")
        f.write(f"  • Median cost: ${np.median(C[C > 0]):.6f}\n\n")
        
        f.write(f"Per-Model Average Costs:\n")
        for j, model in enumerate(models):
            model_costs = C[:, j]
            avg_cost = model_costs[model_costs > 0].mean()
            f.write(f"  • {model:35s}: ${avg_cost:.6f}\n")
    
    print(f"\n✓ Summary report saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Build cost matrix from token statistics and token-based pricing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python tools/build_cost_matrix_from_tokens.py \\
      --pricing config/pricing.yaml \\
      --token-stats reports/token_statistics_final_100pct \\
      --output data/matrices
        """
    )
    
    parser.add_argument('--pricing', type=str, default='config/pricing.yaml',
                       help='Path to pricing.yaml file')
    parser.add_argument('--token-stats', type=str, default='reports/token_statistics_final_100pct',
                       help='Path to token statistics directory')
    parser.add_argument('--output', type=str, default='data/matrices',
                       help='Output directory for cost matrix')
    parser.add_argument('--dataset-dir', type=str, default='.',
                       help='Dataset root directory (for loading meta.parquet)')
    
    args = parser.parse_args()
    
    pricing_file = Path(args.pricing)
    token_stats_dir = Path(args.token_stats)
    output_dir = Path(args.output)
    dataset_dir = Path(args.dataset_dir)
    
    print("\n" + "=" * 80)
    print("🚀 Build Cost Matrix from Token Statistics")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  • Pricing config: {pricing_file}")
    print(f"  • Token stats dir: {token_stats_dir}")
    print(f"  • Output dir: {output_dir}")
    print(f"  • Dataset dir: {dataset_dir}")
    
    # Step 1: Load pricing
    pricing = load_pricing_config(pricing_file)
    
    # Step 2: Load token statistics
    input_df, output_df = load_token_statistics(token_stats_dir)
    
    # Step 2.5: Load valid samples from meta.parquet (to filter out duplicates)
    print("\n" + "=" * 80)
    print("🔍 Step 2.5: Load Valid Samples from Meta")
    print("=" * 80)
    valid_samples = load_valid_samples(dataset_dir)
    
    # Step 3: Calculate sample costs
    cost_df = calculate_sample_costs(input_df, output_df, pricing)
    
    # Step 4: Build cost matrix (returns filtered cost_df)
    C, samples, models, cost_df_filtered = build_cost_matrix(cost_df, input_df, valid_samples)
    
    # Step 5: Save results (use filtered cost_df)
    save_results(C, samples, models, cost_df_filtered, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ Cost Matrix Generation Complete!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  • Cost matrix: {output_dir / 'C.npy'}")
    print(f"  • Sample IDs: {output_dir / 'sample_ids.pkl'}")
    print(f"  • Model names: {output_dir / 'model_names.pkl'}")
    print(f"  • Cost details: {output_dir / 'cost_details.csv'}")
    print(f"  • Summary: {output_dir / 'cost_matrix_summary.txt'}")
    print()


if __name__ == '__main__':
    main()

