#!/usr/bin/env python3
"""
VLM Router Benchmark - Matrix construction script
Build Y (quality) / C (cost) matrices from existing Parquet score files and sample files.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Build Y/C Matrices")
    parser.add_argument('--dataset_dir', default='.', help="Dataset root directory")
    parser.add_argument('--pricing', default='config/pricing.yaml', help="Pricing config file")
    parser.add_argument('--models', default='config/models.yaml', help="Models config file")
    parser.add_argument('--output_dir', default='data', help="Output directory")
    parser.add_argument('--token_costs', default='reports/token_statistics/token_based_costs.csv',
                       help="Token-based costs CSV file (if exists, use token-based costs)")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    
    print("="*80)
    print("🔨 VLM Router - Build Matrices")
    print("="*80)
    
    # 1. Load configuration
    pricing_cfg = load_yaml(args.pricing)
    models_cfg = load_yaml(args.models)
    
    # Check if token-based costs exist
    token_costs_file = Path(args.token_costs)
    use_token_costs = token_costs_file.exists()
    if use_token_costs:
        print(f"\n✓ Found token-based costs file: {token_costs_file}")
        print("  Will use accurate token-based costs instead of estimation")
    else:
        print(f"\n⚠️  Token-based costs not found: {token_costs_file}")
        print("  Will use pricing config estimation")
    
    # 2. Load scoring data
    print("\n📥 Loading scoring data...")
    score_files = list((dataset_dir / "ORACLE" / "score").glob("*.parquet"))
    
    all_scores = []
    for sf in score_files:
        df = pd.read_parquet(sf)
        all_scores.append(df)
        print(f"  ✓ {sf.name}: {len(df)} records")
    
    all_df = pd.concat(all_scores, ignore_index=True)
    print(f"  ✓ Total: {len(all_df)} score records")
    
    # 3. Reshape: one row per sample, one column per model
    print("\n🔄 Reshaping data structure...")
    pivot_quality = all_df.pivot_table(
        index='sample_id',
        columns='model_id',
        values='quality_raw',
        aggfunc='first'
    ).reset_index()
    
    print(f"  ✓ Num samples: {len(pivot_quality)}")
    print(f"  ✓ Model columns: {list(pivot_quality.columns[1:])}")
    
    # 4. Keep only trainable samples
    trainable_ids_file = dataset_dir / "ORACLE" / "trainable_sample_ids.txt"
    if trainable_ids_file.exists():
        with open(trainable_ids_file) as f:
            trainable_ids = set(line.strip() for line in f)
        pivot_quality = pivot_quality[pivot_quality['sample_id'].isin(trainable_ids)]
        print(f"  ✓ Trainable samples: {len(pivot_quality)}")
    
    # 5. Load dataset splits
    print("\n📊 Loading dataset splits...")
    splits = {}
    for split_name in ['train', 'dev', 'test']:
        split_file = dataset_dir / "SPLITS" / f"{split_name}.jsonl"
        if split_file.exists():
            import json
            split_ids = []
            with open(split_file) as f:
                for line in f:
                    split_ids.append(json.loads(line)['sample_id'])
            splits[split_name] = split_ids
            print(f"  ✓ {split_name}: {len(split_ids)} samples")
    
    # 6. Load sample metadata (dataset, task, etc.)
    print("\n📋 Loading sample metadata...")
    import json
    samples = []
    for task_dir in (dataset_dir / "BENCHMARKS").iterdir():
        if not task_dir.is_dir():
            continue
        for samples_file in task_dir.glob("*_samples.jsonl"):
            with open(samples_file) as f:
                for line in f:
                    samples.append(json.loads(line))
    
    sample_meta = {s['sample_id']: s for s in samples}
    print(f"  ✓ Loaded metadata for {len(sample_meta)} samples")
    
    # 7. Build matrices
    print("\n🏗️  Building Y/C matrices...")
    
    # Model list
    model_cols = [col for col in pivot_quality.columns if col != 'sample_id']
    model_list = sorted(model_cols)
    K = len(model_list)
    
    # Sample ID list
    sample_ids = pivot_quality['sample_id'].tolist()
    N = len(sample_ids)
    
    print(f"  N={N} samples, K={K} models")
    
    # Y matrix (N x K): quality_raw (0/1)
    Y = np.zeros((N, K), dtype=np.int8)
    for i, sid in enumerate(sample_ids):
        row = pivot_quality[pivot_quality['sample_id'] == sid].iloc[0]
        for j, model in enumerate(model_list):
            Y[i, j] = int(row[model])
    
    # C matrix (N x K): cost ($/sample)
    C = np.zeros((N, K), dtype=np.float32)
    for j, model in enumerate(model_list):
        cost = pricing_cfg['pricing'].get(model, {}).get('usd_per_sample', 0.01)
        C[:, j] = cost
    
    print(f"  ✓ Y matrix: {Y.shape}, mean accuracy: {Y.mean():.4f}")
    print(f"  ✓ C matrix: {C.shape}, mean cost: ${C.mean():.6f}")
    
    # 8. Save matrices
    print("\n💾 Saving matrices and indices...")
    matrices_dir = output_dir / "matrices"
    registry_dir = output_dir / "registry"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(matrices_dir / 'Y.npz', Y=Y)
    np.savez_compressed(matrices_dir / 'C.npz', C=C)
    
    with open(registry_dir / 'sample_index.pkl', 'wb') as f:
        pickle.dump(sample_ids, f)
    
    with open(registry_dir / 'model_index.pkl', 'wb') as f:
        pickle.dump(model_list, f)
    
    # Save metadata
    meta_rows = []
    for sid in sample_ids:
        meta = sample_meta.get(sid, {})
        # Determine split
        split = None
        for split_name, split_list in splits.items():
            if sid in split_list:
                split = split_name
                break
        
        meta_rows.append({
            'sample_id': sid,
            'dataset': meta.get('dataset', 'unknown'),
            'split': split or 'unknown',
            'task': meta.get('task_type', 'unknown')
        })
    
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(registry_dir / 'meta.parquet', index=False)
    
    print(f"  ✓ Y matrix: {matrices_dir / 'Y.npz'}")
    print(f"  ✓ C matrix: {matrices_dir / 'C.npz'}")
    print(f"  ✓ Sample index: {registry_dir / 'sample_index.pkl'}")
    print(f"  ✓ Model index: {registry_dir / 'model_index.pkl'}")
    print(f"  ✓ Metadata: {registry_dir / 'meta.parquet'}")
    
    # 9. Summary stats
    print("\n📈 Summary:")
    print(f"  Models: {model_list}")
    for j, model in enumerate(model_list):
        acc = Y[:, j].mean()
        cost = C[0, j]
        print(f"    {model}: accuracy={acc:.4f}, cost=${cost:.6f}/sample")
    
    print("\n  By split:")
    for split_name, split_list in splits.items():
        split_indices = [i for i, sid in enumerate(sample_ids) if sid in split_list]
        if split_indices:
            split_Y = Y[split_indices]
            print(f"    {split_name}: {len(split_indices)} samples")
            for j, model in enumerate(model_list):
                acc = split_Y[:, j].mean()
                print(f"      {model}: {acc:.4f}")
    
    print("\n" + "="*80)
    print("✅ Matrix construction complete!")
    print("="*80)


if __name__ == '__main__':
    main()


