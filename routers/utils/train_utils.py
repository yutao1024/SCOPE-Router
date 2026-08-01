#!/usr/bin/env python3
"""
Training utilities - shared data loading and evaluation logic.
Used by various predictive routers.
"""

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


def model_name_to_filename(model_name: str) -> str:
    """
    Convert a model name to a filename (parquet file stem).
    
    Examples:
        'BAAI/bge-m3' -> 'bge-m3'
        'facebook/dinov2-base' -> 'dinov2-base'
        'intfloat/e5-large-v2' -> 'e5-large-v2'
    """
    # Remove org/user prefix, keep model name
    if '/' in model_name:
        filename = model_name.split('/')[-1]
    else:
        filename = model_name
    
    return filename


def load_data_for_training(
    dataset_dir: Path,
    text_encoder: str = "BAAI/bge-m3",
    vision_encoder: str = "facebook/dinov2-base",
    query_embedding_file: str = None,
):
    """
    Load all data required for training.
    
    Args:
        dataset_dir: Dataset root directory
        text_encoder: Text encoder model name (default: "BAAI/bge-m3")
        vision_encoder: Vision encoder model name (default: "facebook/dinov2-base")
    
    Returns:
        dict containing: Y, C, meta, models, text_embeddings, vision_embeddings, sample_ids, splits,
        and the encoder names
    """
    print("📥 Loading data...")
    if query_embedding_file:
        print(f"  Query embedding file: {query_embedding_file}")
    else:
        print(f"  Text encoder: {text_encoder}")
        print(f"  Vision encoder: {vision_encoder}")
    
    # Load Y matrix
    Y = np.load(dataset_dir / "data/matrices/Y.npz")['Y']
    
    # Load metadata
    meta = pd.read_parquet(dataset_dir / "data/registry/meta.parquet")
    
    # Load cost matrix C - prefer token-based (C.npy), otherwise legacy (C.npz)
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
    
    # Load model index
    with open(dataset_dir / "data/registry/model_index.pkl", 'rb') as f:
        models = pickle.load(f)
    
    query_embeddings = None
    text_embeddings = None
    vision_embeddings = None
    meta_sample_order = meta.set_index('sample_id').index

    if query_embedding_file:
        query_embedding_path = Path(query_embedding_file)
        if not query_embedding_path.is_absolute():
            query_embedding_path = dataset_dir / query_embedding_path
        if not query_embedding_path.exists():
            raise FileNotFoundError(f"Query embedding file not found: {query_embedding_path}")

        print(f"  Loading query embeddings: {query_embedding_path}")
        query_df = pd.read_parquet(query_embedding_path)
        if "sample_id" not in query_df.columns or "embedding" not in query_df.columns:
            raise ValueError(f"{query_embedding_path} must contain columns: sample_id, embedding")
        embeddings_df = query_df[["sample_id", "embedding"]].merge(meta[["sample_id"]], on="sample_id")
        embeddings_df = embeddings_df.set_index("sample_id").reindex(meta_sample_order).reset_index()
        if embeddings_df["embedding"].isna().any():
            missing = embeddings_df.loc[embeddings_df["embedding"].isna(), "sample_id"].head().tolist()
            raise ValueError(f"Missing query embeddings for meta samples, preview: {missing}")
        query_embeddings = np.vstack(embeddings_df["embedding"].values).astype(np.float32)
        sample_ids = embeddings_df["sample_id"].values
    else:
        # Convert model names to filenames
        text_filename = model_name_to_filename(text_encoder)
        vision_filename = model_name_to_filename(vision_encoder)

        # Load text and vision embeddings
        text_embedding_path = dataset_dir / "EMBEDDINGS/text" / f"{text_filename}.parquet"
        vision_embedding_path = dataset_dir / "EMBEDDINGS/vision" / f"{vision_filename}.parquet"

        if not text_embedding_path.exists():
            raise FileNotFoundError(
                f"Text embedding file not found: {text_embedding_path}\n"
                f"Please extract features for {text_encoder}, or use --text_encoder to select another model"
            )

        if not vision_embedding_path.exists():
            raise FileNotFoundError(
                f"Vision embedding file not found: {vision_embedding_path}\n"
                f"Please extract features for {vision_encoder}, or use --vision_encoder to select another model"
            )

        print(f"  Loading text embeddings: {text_embedding_path.name}")
        text_df = pd.read_parquet(text_embedding_path)

        print(f"  Loading vision embeddings: {vision_embedding_path.name}")
        vision_df = pd.read_parquet(vision_embedding_path)

        # Merge embeddings, ensuring sample_id matches
        embeddings_df = text_df.merge(vision_df, on='sample_id', suffixes=('_text', '_vision'))
        embeddings_df = embeddings_df.merge(meta[['sample_id']], on='sample_id')

        # Reorder to match meta order
        embeddings_df = embeddings_df.set_index('sample_id').reindex(meta_sample_order).reset_index()

        # Extract arrays
        text_embeddings = np.vstack(embeddings_df['embedding_text'].values)
        vision_embeddings = np.vstack(embeddings_df['embedding_vision'].values)
        sample_ids = embeddings_df['sample_id'].values
    
    print(f"  Matched samples: {len(sample_ids)}")
    
    # Load splits
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
    print(f"  Models: {models}")
    if query_embeddings is not None:
        print(f"  Query embeddings: {query_embeddings.shape}")
    else:
        print(f"  Text embeddings: {text_embeddings.shape}")
        print(f"  Vision embeddings: {vision_embeddings.shape}")
    print(f"  Splits: train={len(splits.get('train', []))}, "
          f"dev={len(splits.get('dev', []))}, test={len(splits.get('test', []))}")
    
    return {
        'Y': Y,
        'C': C,
        'meta': meta,
        'models': models,
        'query_embeddings': query_embeddings,
        'text_embeddings': text_embeddings,
        'vision_embeddings': vision_embeddings,
        'sample_ids': sample_ids,
        'splits': splits,
        'text_encoder': text_encoder,
        'vision_encoder': vision_encoder,
        'query_embedding_file': str(query_embedding_file) if query_embedding_file else None,
    }


def align_train_data(data, train_ids):
    """
    Align training data to ensure meta, embeddings, and train split match.
    
    Returns:
        (X_text_train, X_vision_train, Y_train, C_train, meta_train,
         aligned_meta_indices, aligned_embedding_indices)
    """
    meta_sample_ids = data['meta']['sample_id'].values
    embeddings_sample_ids = data['sample_ids']
    
    # Find samples in train split
    meta_train_mask = pd.Series(meta_sample_ids).isin(train_ids)
    embeddings_train_mask = pd.Series(embeddings_sample_ids).isin(train_ids)
    
    train_meta_indices = np.where(meta_train_mask.values)[0]
    train_embedding_indices = np.where(embeddings_train_mask.values)[0]
    
    # Align via sample_id
    train_meta_ids = meta_sample_ids[train_meta_indices]
    train_embedding_ids = embeddings_sample_ids[train_embedding_indices]
    
    # Find common samples
    common_train_ids = sorted(set(train_meta_ids) & set(train_embedding_ids))
    
    # Re-index to align
    meta_id_to_idx = {sid: idx for idx, sid in enumerate(meta_sample_ids)}
    embedding_id_to_idx = {sid: idx for idx, sid in enumerate(embeddings_sample_ids)}
    
    aligned_meta_indices = np.array([meta_id_to_idx[sid] for sid in common_train_ids])
    aligned_embedding_indices = np.array([embedding_id_to_idx[sid] for sid in common_train_ids])
    
    # Extract aligned arrays. In joint-query mode, X_text_train carries the
    # single query embedding and X_vision_train is None for backward-compatible
    # tuple shape.
    if data.get('query_embeddings') is not None:
        X_text_train = data['query_embeddings'][aligned_embedding_indices]
        X_vision_train = None
    else:
        X_text_train = data['text_embeddings'][aligned_embedding_indices]
        X_vision_train = data['vision_embeddings'][aligned_embedding_indices]
    Y_train = data['Y'][aligned_meta_indices]
    C_train = data['C'][aligned_meta_indices]
    meta_train = data['meta'].iloc[aligned_meta_indices].copy()
    
    return (X_text_train, X_vision_train, Y_train, C_train, meta_train,
            aligned_meta_indices, aligned_embedding_indices)


def evaluate_router_with_indices(router, Y_test, C_test, meta_test, models,
                                   text_embeddings, vision_embeddings, embedding_indices,
                                   cost_bounds_file=None, query_embeddings=None):
    """
    Evaluate router performance (using specified embedding indices).
    
    Args:
        cost_bounds_file: Optional path to cost_bounds.json for Rank Score calculation
    
    Returns:
        dict containing: accuracy, avg_cost, num_correct, num_samples, rank_score (if cost_bounds provided)
    """
    # Slice test embeddings and predict.
    if query_embeddings is not None:
        X_test = query_embeddings[embedding_indices]
        predictions = router.predict(X=X_test)
    else:
        X_text_test = text_embeddings[embedding_indices]
        X_vision_test = vision_embeddings[embedding_indices]
        predictions = router.predict(X_text=X_text_test, X_vision=X_vision_test)
    
    # Convert to model indices
    if hasattr(router, 'reverse_mapping') and router.reverse_mapping:
        model_indices = [router.reverse_mapping.get(router.model_mapping.get(p, p), p) 
                         for p in predictions]
    else:
        model_indices = predictions
    model_indices = np.array(model_indices)
    
    # Compute accuracy and cost
    correct = Y_test[np.arange(len(model_indices)), model_indices]
    accuracy = correct.mean()
    
    costs = C_test[np.arange(len(model_indices)), model_indices]
    avg_cost = costs.mean()
    
    num_correct = int(correct.sum())
    num_samples = len(correct)
    
    result = {
        'accuracy': float(accuracy),
        'avg_cost': float(avg_cost),
        'num_correct': int(num_correct),
        'num_samples': int(num_samples)
    }
    
    # Calculate Rank Score if cost bounds provided
    if cost_bounds_file is not None:
        try:
            from routers.utils.rank_score import rank_score, get_cost_bounds_from_config
            cmin, cmax = get_cost_bounds_from_config(cost_bounds_file)
            score = rank_score(accuracy, avg_cost, cmin, cmax, beta=0.1)
            result['rank_score'] = float(score)
            result['rank_score_beta'] = 0.1
        except Exception as e:
            print(f"  Warning: Could not calculate Rank Score: {e}")
    
    return result


def profile_router_latency(router, X_text=None, X_vision=None, meta=None, batch_size=16,
                           warmup_runs=5, test_runs=50, X=None):
    """
    Profile router inference latency at specified batch size
    
    Args:
        router: Router instance to profile
        X_text: Text embeddings
        X_vision: Vision embeddings
        meta: Metadata DataFrame (optional, for token-based metrics)
        batch_size: Batch size to test (default: 16)
        warmup_runs: Number of warmup iterations (default: 5)
        test_runs: Number of test iterations (default: 50)
    
    Returns:
        dict with latency metrics:
            - ms_per_sample: milliseconds per sample
            - throughput: samples per second
            - ms_per_token: milliseconds per token (if meta provided)
            - throughput_tokens: tokens per second (if meta provided)
            - total_time_ms: total inference time in milliseconds
            - batch_size: batch size used
            - token_counting_method: method used for token counting (if meta provided)
    """
    # Allow fast/CI-friendly overrides via environment variables
    # - VLM_ROUTER_SKIP_LATENCY=1 : skip profiling entirely
    # - VLM_ROUTER_LATENCY_WARMUP_RUNS=... : override warmup runs
    # - VLM_ROUTER_LATENCY_TEST_RUNS=... : override test runs
    try:
        import os
        skip_latency = os.getenv("VLM_ROUTER_SKIP_LATENCY", "0").strip().lower() in ("1", "true", "yes")
        if skip_latency:
            return {
                'ms_per_sample': -1,
                'throughput': -1,
                'total_time_ms': -1,
                'batch_size': int(batch_size),
                'skipped': True,
                'skip_reason': 'VLM_ROUTER_SKIP_LATENCY=1'
            }

        warmup_env = os.getenv("VLM_ROUTER_LATENCY_WARMUP_RUNS")
        test_env = os.getenv("VLM_ROUTER_LATENCY_TEST_RUNS")
        if warmup_env is not None and str(warmup_env).strip() != "":
            warmup_runs = int(warmup_env)
        if test_env is not None and str(test_env).strip() != "":
            test_runs = int(test_env)
    except Exception:
        # Never fail profiling because of env parsing
        pass

    try:
        from routers.utils.latency_profiler import LatencyProfiler
        
        profiler = LatencyProfiler(warmup_runs=warmup_runs, test_runs=test_runs)
        
        # Profile at specified batch size
        results = profiler.profile_router(
            router=router,
            X=X,
            X_text=X_text,
            X_vision=X_vision,
            meta=meta,
            batch_sizes=[batch_size]
        )
        
        # Extract results for the specified batch size
        batch_results = results['batch_results'].get(batch_size, {})
        
        if batch_results:
            # LatencyProfiler returns nested dicts with 'mean', 'std', etc.
            metrics = {
                'ms_per_sample': float(batch_results['per_sample_ms']['mean']),
                'throughput': float(batch_results['throughput_samples_per_sec']),
                'total_time_ms': float(batch_results['total_time_ms']['mean']),
                'batch_size': int(batch_size)
            }
            
            # Add per-token metrics if available
            if 'per_token_ms' in batch_results:
                metrics['ms_per_token'] = float(batch_results['per_token_ms']['mean'])
                metrics['throughput_tokens'] = float(batch_results['throughput_tokens_per_sec'])
                metrics['token_counting_method'] = results.get('token_counting_method', 'unknown')
            
            return metrics
        else:
            return {
                'ms_per_sample': -1,
                'throughput': -1,
                'total_time_ms': -1,
                'batch_size': batch_size
            }
    
    except Exception as e:
        print(f"  Warning: Could not profile latency: {e}")
        return {
            'ms_per_sample': -1,
            'throughput': -1,
            'total_time_ms': -1,
            'batch_size': batch_size
        }
