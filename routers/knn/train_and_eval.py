#!/usr/bin/env python3
"""
KNN
，baselines

Usage:
    python routers/knn/train_and_eval.py \
        --dataset_dir . \
        --k 5 \
        --fusion_method concat \
        --output_dir outputs/knn_router
"""

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# NOTE: (translated from Chinese)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routers.knn.router import KNNRouter
from routers.common import quality, cost
from routers.utils.fusion import fuse_embeddings
from routers.utils.train_utils import evaluate_router_with_indices, profile_router_latency
from routers.utils.rank_score import rank_score, get_cost_bounds_from_config


def model_name_to_filename(model_name: str) -> str:
    """"""
    if '/' in model_name:
        return model_name.split('/')[-1]
    return model_name


def evaluate_by_dataset(router, Y_split, C_split, meta_split, models, 
                        text_embeddings, vision_embeddings, split_embedding_indices,
                        cmin, cmax, beta=0.1):
    """
    
    
    Args:
        router: 
        Y_split:  [num_samples, num_models]
        C_split:  [num_samples, num_models]
        meta_split:  DataFrame
        models: 
        text_embeddings:  [num_total_samples, dim]
        vision_embeddings:  [num_total_samples, dim]
        split_embedding_indices: splitembeddings
        cmin: （ rank_score）
        cmax: （ rank_score）
        beta: rank_score  beta 
    
    Returns:
        DataFrame with columns: dataset, accuracy, avg_cost, rank_score, num_correct, num_samples
    """
    print(f"\n📊 evaluation...")
    
    #  dataset 
    if 'dataset' not in meta_split.columns:
        print(f"  ⚠️  No 'dataset' column found in metadata")
        return pd.DataFrame()
    
    # NOTE: (translated from Chinese)
    X_text = text_embeddings[split_embedding_indices]
    X_vision = vision_embeddings[split_embedding_indices]
    
    # split
    predictions = router.predict(X_text=X_text, X_vision=X_vision, meta=meta_split)
    
    # NOTE: (translated from Chinese)
    datasets = meta_split['dataset'].unique()
    print(f"  ✓ Found {len(datasets)} datasets")
    
    results = []
    
    for dataset_name in sorted(datasets):
        # （meta_split）
        dataset_mask = (meta_split['dataset'] == dataset_name).values
        dataset_indices = np.where(dataset_mask)[0]
        
        if len(dataset_indices) == 0:
            continue
        
        # 、、
        dataset_preds = predictions[dataset_indices]
        dataset_Y = Y_split[dataset_indices]
        dataset_C = C_split[dataset_indices]
        
        # NOTE: (translated from Chinese)
        correct = 0
        total_cost = 0.0
        
        for i, pred_idx in enumerate(dataset_preds):
            if 0 <= pred_idx < dataset_Y.shape[1]:
                # NOTE: (translated from Chinese)
                if dataset_Y[i, pred_idx] == 1:
                    correct += 1
                # NOTE: (translated from Chinese)
                total_cost += dataset_C[i, pred_idx]
        
        num_samples = len(dataset_indices)
        accuracy = correct / num_samples if num_samples > 0 else 0.0
        avg_cost = total_cost / num_samples if num_samples > 0 else 0.0
        
        #  rank_score
        rs = rank_score(accuracy, avg_cost, cmin, cmax, beta)
        
        results.append({
            'dataset': dataset_name,
            'accuracy': accuracy,
            'avg_cost': avg_cost,
            'rank_score': rs,
            'num_correct': correct,
            'num_samples': num_samples
        })
    
    results_df = pd.DataFrame(results).sort_values('rank_score', ascending=False)
    
    print(f"  ✓ Evaluated {len(results_df)} datasets")
    
    return results_df


def load_data(dataset_dir: Path, text_encoder: str = "BAAI/bge-m3", vision_encoder: str = "facebook/dinov2-base"):
    """load"""
    print("📥 load...")
    print(f"  Text encoder: {text_encoder}")
    print(f"  Vision encoder: {vision_encoder}")
    
    # Y
    Y = np.load(dataset_dir / "data/matrices/Y.npz")['Y']
    
    # NOTE: (translated from Chinese)
    meta = pd.read_parquet(dataset_dir / "data/registry/meta.parquet")
    
    #  C - token-based (C.npy)，(C.npz)
    cost_npy = dataset_dir / "data/matrices/C.npy"
    cost_npz = dataset_dir / "data/matrices/C.npz"
    
    if cost_npy.exists():
        # Load token-based cost matrix
        C_full = np.load(cost_npy)
        
        # Load sample IDs for cost matrix
        sample_ids_file = dataset_dir / "data/matrices/sample_ids.pkl"
        if sample_ids_file.exists():
            with open(sample_ids_file, 'rb') as f:
                c_sample_ids = pickle.load(f)
            
            # Align C with meta by sample_id
            meta_sample_ids = meta['sample_id'].values
            
            # Create a mapping from sample_id to index in C_full
            c_id_to_idx = {sid: i for i, sid in enumerate(c_sample_ids)}
            
            # Create aligned C matrix
            C = np.zeros((len(meta), C_full.shape[1]), dtype=C_full.dtype)
            missing_count = 0
            for i, sid in enumerate(meta_sample_ids):
                if sid in c_id_to_idx:
                    C[i] = C_full[c_id_to_idx[sid]]
                else:
                    # If sample not found, use column means as fallback
                    C[i] = C_full.mean(axis=0)
                    missing_count += 1
            
            if missing_count > 0:
                print(f"  ⚠️  {missing_count} samples not found in cost matrix, used mean values")
            
            print("  ✓ Using token-based cost matrix (C.npy), aligned with meta")
        else:
            # If no sample_ids file, assume same order
            if len(C_full) == len(meta):
                C = C_full
                print("  ✓ Using token-based cost matrix (C.npy)")
            else:
                raise ValueError(f"Cost matrix size ({len(C_full)}) doesn't match meta size ({len(meta)}) and no sample_ids.pkl found")
    
    elif cost_npz.exists():
        C = np.load(cost_npz)['C']
        print("  ⚠️  Using legacy cost matrix (C.npz) - consider regenerating with token-based costs")
    else:
        raise FileNotFoundError(f"Cost matrix not found in {dataset_dir / 'data/matrices/'}")
    
    # NOTE: (translated from Chinese)
    with open(dataset_dir / "data/registry/model_index.pkl", 'rb') as f:
        models = pickle.load(f)
    
    # NOTE: (translated from Chinese)
    text_filename = model_name_to_filename(text_encoder)
    vision_filename = model_name_to_filename(vision_encoder)
    
    # NOTE: (translated from Chinese)
    text_embedding_path = dataset_dir / "EMBEDDINGS/text" / f"{text_filename}.parquet"
    vision_embedding_path = dataset_dir / "EMBEDDINGS/vision" / f"{vision_filename}.parquet"
    
    if not text_embedding_path.exists():
        raise FileNotFoundError(
            f": {text_embedding_path}\n"
            f" {text_encoder} ， --text_encoder "
        )
    
    if not vision_embedding_path.exists():
        raise FileNotFoundError(
            f": {vision_embedding_path}\n"
            f" {vision_encoder} ， --vision_encoder "
        )
    
    print(f"  load: {text_embedding_path.name}")
    text_df = pd.read_parquet(text_embedding_path)
    
    print(f"  load: {vision_embedding_path.name}")
    vision_df = pd.read_parquet(vision_embedding_path)
    
    # ，sample_id
    embeddings_df = text_df.merge(vision_df, on='sample_id', suffixes=('_text', '_vision'))
    embeddings_df = embeddings_df.merge(meta[['sample_id']], on='sample_id')
    
    # meta
    meta_sample_order = meta.set_index('sample_id').index
    embeddings_df = embeddings_df.set_index('sample_id').reindex(meta_sample_order).reset_index()
    
    # NOTE: (translated from Chinese)
    text_embeddings = np.vstack(embeddings_df['embedding_text'].values)
    vision_embeddings = np.vstack(embeddings_df['embedding_vision'].values)
    sample_ids = embeddings_df['sample_id'].values
    
    print(f"  samples: {len(sample_ids)}")
    
    # NOTE: (translated from Chinese)
    splits = {}
    for split_name in ['train', 'dev', 'test']:
        split_file = dataset_dir / "SPLITS" / f"{split_name}.jsonl"
        if split_file.exists():
            split_ids = []
            with open(split_file) as f:
                for line in f:
                    split_ids.append(json.loads(line)['sample_id'])
            splits[split_name] = split_ids
    
    print(f"  Y: {Y.shape}, C: {C.shape}")
    print(f"  : {models}")
    print(f"  : {text_embeddings.shape}")
    print(f"  : {vision_embeddings.shape}")
    print(f"  : train={len(splits.get('train', []))}, "
          f"dev={len(splits.get('dev', []))}, test={len(splits.get('test', []))}")
    
    return {
        'Y': Y,
        'C': C,
        'meta': meta,
        'models': models,
        'text_embeddings': text_embeddings,
        'vision_embeddings': vision_embeddings,
        'sample_ids': sample_ids,
        'splits': splits
    }


def main():
    parser = argparse.ArgumentParser(description="training and evaluationKNN")
    parser.add_argument("--dataset_dir", default=".", help="Dataset root directory")
    parser.add_argument("--k", type=int, default=5, help="KNNK")
    parser.add_argument("--fusion_method", default="concat",
                       choices=['concat', 'average', 'weighted_average', 'normalize_concat'],
                       help="")
    parser.add_argument("--text_weight", type=float, default=0.5,
                       help="Text weight（weighted_average）")
    parser.add_argument("--text_encoder", default="BAAI/bge-m3",
                       help="Text encoder（: BAAI/bge-m3）")
    parser.add_argument("--vision_encoder", default="facebook/dinov2-base",
                       help="Vision encoder（: facebook/dinov2-base）")
    parser.add_argument("--output_dir", default="outputs/knn_router", help="Output directory")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🚀 KNNtraining and evaluation")
    print("="*80)
    print(f"K: {args.k}")
    print(f"Fusion method: {args.fusion_method}")
    if args.fusion_method == 'weighted_average':
        print(f"Text weight: {args.text_weight}")
    
    # 1. 
    data = load_data(dataset_dir, text_encoder=args.text_encoder, vision_encoder=args.vision_encoder)
    
    # Load cost bounds for Rank Score calculation
    cost_bounds_file = dataset_dir / "data/matrices/cost_bounds.json"
    if not cost_bounds_file.exists():
        cost_bounds_file = None
        print("  ⚠️  cost_bounds.json not found, Rank Score will not be calculated")
    
    # 2. （）
    print("\n🔧 Preparing training data...")
    train_ids = set(data['splits'].get('train', []))
    
    # meta、embeddingstrain split
    # NOTE: (translated from Chinese)
    meta_sample_ids = data['meta']['sample_id'].values
    embeddings_sample_ids = data['sample_ids']
    
    # train splitID
    meta_train_mask = pd.Series(meta_sample_ids).isin(train_ids)
    embeddings_train_mask = pd.Series(embeddings_sample_ids).isin(train_ids)
    
    # （metaembeddings）
    train_meta_indices = np.where(meta_train_mask.values)[0]
    train_embedding_indices = np.where(embeddings_train_mask.values)[0]
    
    # sample_id
    train_meta_ids = meta_sample_ids[train_meta_indices]
    train_embedding_ids = embeddings_sample_ids[train_embedding_indices]
    
    # NOTE: (translated from Chinese)
    common_train_ids = set(train_meta_ids) & set(train_embedding_ids)
    common_train_ids = sorted(common_train_ids)  # 
    
    # NOTE: (translated from Chinese)
    meta_id_to_idx = {sid: idx for idx, sid in enumerate(meta_sample_ids)}
    embedding_id_to_idx = {sid: idx for idx, sid in enumerate(embeddings_sample_ids)}
    
    aligned_meta_indices = np.array([meta_id_to_idx[sid] for sid in common_train_ids])
    aligned_embedding_indices = np.array([embedding_id_to_idx[sid] for sid in common_train_ids])
    
    # NOTE: (translated from Chinese)
    X_text_train = data['text_embeddings'][aligned_embedding_indices]
    X_vision_train = data['vision_embeddings'][aligned_embedding_indices]
    Y_train = data['Y'][aligned_meta_indices]
    C_train = data['C'][aligned_meta_indices]
    meta_train = data['meta'].iloc[aligned_meta_indices].copy()
    
    print(f"  Train set: {len(common_train_ids)} samples（）")
    
    # 3. KNN
    print("\n🎓 trainingKNN...")
    router = KNNRouter(
        n_neighbors=args.k,
        fusion_method=args.fusion_method,
        text_weight=args.text_weight,
        text_encoder=args.text_encoder,
        vision_encoder=args.vision_encoder
    )
    
    # NOTE: (translated from Chinese)
    K = Y_train.shape[1]
    model_mapping = {i: data['models'][i] for i in range(K)}
    
    router.fit(Y_train, C_train, meta_train,
               X_text=X_text_train, X_vision=X_vision_train,
               model_mapping=model_mapping)
    
    # 4. （）
    model_name = f"knn_k{args.k}_{args.fusion_method}"
    if args.fusion_method == 'weighted_average':
        model_name += f"_w{args.text_weight:.2f}"
    
    model_path = output_dir / f"{model_name}.pkl"
    router.save(str(model_path))
    print(f"\n✓ Model saved: {model_path}")
    
    # 5. 
    print("\n📊 evaluation...")
    results = {}
    
    for split_name in ['dev', 'test']:
        if split_name not in data['splits']:
            continue
        
        split_ids = set(data['splits'][split_name])
        
        # splitmeta
        meta_sample_ids = data['meta']['sample_id'].values
        split_mask = pd.Series(meta_sample_ids).isin(split_ids)
        split_indices = np.where(split_mask.values)[0]
        
        # splitembeddings
        embeddings_sample_ids = data['sample_ids']
        split_embedding_mask = pd.Series(embeddings_sample_ids).isin(split_ids)
        split_embedding_indices = np.where(split_embedding_mask)[0]
        
        if len(split_indices) == 0:
            continue
        
        Y_split = data['Y'][split_indices]
        C_split = data['C'][split_indices]
        meta_split = data['meta'].iloc[split_indices].copy()
        
        # evaluate_router
        # meta_splitembeddings
        result = evaluate_router_with_indices(
            router, Y_split, C_split, meta_split, data['models'],
            data['text_embeddings'], data['vision_embeddings'], split_embedding_indices,
            cost_bounds_file=cost_bounds_file
        )
        results[split_name] = result
        
        print(f"  {split_name}:")
        print(f"    Accuracy: {result['accuracy']:.4f} ({result['num_correct']}/{result['num_samples']})")
        print(f"    Avg cost: ${result['avg_cost']:.6f}")
        if 'rank_score' in result:
            print(f"    Rank Score: {result['rank_score']:.4f}")
    
    # 6.  test set
    dataset_results = {}
    if 'test' in data['splits']:
        #  cost bounds（ rank_score ）
        try:
            cmin, cmax = get_cost_bounds_from_config(str(cost_bounds_file))
            print(f"\n✓ Using cost bounds: cmin=${cmin:.6f}, cmax=${cmax:.6f}")
        except Exception as e:
            # NOTE: (translated from Chinese)
            cmin = data['C'].min()
            cmax = data['C'].max()
            print(f"\n✓ Calculated cost bounds from data: cmin=${cmin:.6f}, cmax=${cmax:.6f}")
        
        #  test set 
        test_ids = set(data['splits']['test'])
        test_meta_mask = data['meta']['sample_id'].isin(test_ids)
        test_meta_indices = np.where(test_meta_mask.values)[0]
        test_embedding_mask = pd.Series(data['sample_ids']).isin(test_ids)
        test_embedding_indices = np.where(test_embedding_mask)[0]
        
        Y_test = data['Y'][test_meta_indices]
        C_test = data['C'][test_meta_indices]
        meta_test = data['meta'].iloc[test_meta_indices].copy()
        
        dataset_results_df = evaluate_by_dataset(
            router=router,
            Y_split=Y_test,
            C_split=C_test,
            meta_split=meta_test,
            models=data['models'],
            text_embeddings=data['text_embeddings'],
            vision_embeddings=data['vision_embeddings'],
            split_embedding_indices=test_embedding_indices,
            cmin=cmin,
            cmax=cmax,
            beta=0.1
        )
        
        if not dataset_results_df.empty:
            # NOTE: (translated from Chinese)
            dataset_results = dataset_results_df.to_dict('records')
            
            #  top 10  bottom 10 
            print(f"\n  📈 Top 10 datasets by rank_score:")
            for i, row in dataset_results_df.head(10).iterrows():
                print(f"    {row['dataset']:<30} Acc: {row['accuracy']:.4f}, "
                      f"Cost: ${row['avg_cost']:.6f}, RS: {row['rank_score']:.4f}, "
                      f"Samples: {row['num_samples']}")
            
            print(f"\n  📉 Bottom 10 datasets by rank_score:")
            for i, row in dataset_results_df.tail(10).iterrows():
                print(f"    {row['dataset']:<30} Acc: {row['accuracy']:.4f}, "
                      f"Cost: ${row['avg_cost']:.6f}, RS: {row['rank_score']:.4f}, "
                      f"Samples: {row['num_samples']}")
    
    # 7. Profile latency on test set
    print("\n⏱️  Profiling latency...")
    test_ids = set(data['splits']['test'])
    test_embedding_mask = pd.Series(data['sample_ids']).isin(test_ids)
    test_embedding_indices = np.where(test_embedding_mask)[0]
    X_text_test = data['text_embeddings'][test_embedding_indices]
    X_vision_test = data['vision_embeddings'][test_embedding_indices]
    
    # Get meta for test set
    test_meta_mask = data['meta']['sample_id'].isin(test_ids)
    test_meta_indices = np.where(test_meta_mask.values)[0]
    meta_test = data['meta'].iloc[test_meta_indices]
    
    latency_metrics = profile_router_latency(
        router=router,
        X_text=X_text_test,
        X_vision=X_vision_test,
        meta=meta_test,
        batch_size=16,
        warmup_runs=5,
        test_runs=50
    )
    
    print(f"  Batch size: {latency_metrics['batch_size']}")
    if 'ms_per_token' in latency_metrics:
        print(f"  Per-token latency: {latency_metrics['ms_per_token']:.4f} ms/token")
        print(f"  Token throughput: {latency_metrics['throughput_tokens']:.1f} tokens/sec")
        print(f"  Token counting: {latency_metrics.get('token_counting_method', 'unknown')}")
    print(f"  Per-sample latency: {latency_metrics['ms_per_sample']:.3f} ms/sample")
    print(f"  Sample throughput: {latency_metrics['throughput']:.1f} samples/sec")
    
    # 8. 
    report = {
        'model': model_name,
        'hyperparameters': {
            'k': args.k,
            'fusion_method': args.fusion_method,
            'text_weight': args.text_weight if args.fusion_method == 'weighted_average' else None
        },
        'results': results,
        'results_by_dataset': dataset_results,
        'latency': latency_metrics
    }
    
    report_path = output_dir / f"{model_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved: {report_path}")
    
    #  CSV 
    if dataset_results:
        dataset_csv_path = output_dir / f"{model_name}_test_by_dataset.csv"
        dataset_results_df.to_csv(dataset_csv_path, index=False, float_format='%.6f')
        print(f"✓ Per-dataset results saved: {dataset_csv_path}")
    
    # 9. CSV（baselines）
    test_results = results.get('test', {})
    summary_row = {
        'router': model_name,
        'accuracy': test_results.get('accuracy', 0),
        'avg_quality': test_results.get('accuracy', 0),
        'avg_cost': test_results.get('avg_cost', 0),
        'num_samples': test_results.get('num_samples', 0),
        'num_correct': test_results.get('num_correct', 0),
        'rank_score': test_results.get('rank_score', -1),
        'latency_ms_per_sample': latency_metrics.get('ms_per_sample', -1),
        'throughput_samples_per_sec': latency_metrics.get('throughput', -1),
        'latency_ms_per_token': latency_metrics.get('ms_per_token', -1),
        'throughput_tokens_per_sec': latency_metrics.get('throughput_tokens', -1),
        'token_counting_method': latency_metrics.get('token_counting_method', 'N/A')
    }
    
    summary_path = output_dir / f"{model_name}_summary.csv"
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    print(f"✓ Summary saved: {summary_path}")
    
    print("\n" + "="*80)
    print("✅ training and evaluation！")
    print("="*80)
    
    return report


if __name__ == "__main__":
    main()

