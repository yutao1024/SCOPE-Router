#!/usr/bin/env python3
"""
Latency Profiler for VLM Routers

Measures inference latency:
- Per-token latency (ms/token, tokens/s) - primary metric
- Per-sample latency (ms/sample, samples/s) - secondary metric

Uses accurate token counting when available (tiktoken + model-specific image-token estimates).
"""

import time
import numpy as np
from typing import Dict, Optional, List, Tuple
import torch
import warnings


def estimate_tokens_from_meta(meta, tokenizer=None) -> np.ndarray:
    """
    Estimate tokens per sample from a meta DataFrame.

    Prefer routers.utils.token_stats.TokenCounter (accurate tiktoken + image-token estimates).
    Fallback: simple estimation.
    
    Args:
        meta: Metadata DataFrame
        tokenizer: Optional tokenizer (fallback only)
    
    Returns:
        Token count array (N,)
    """
    try:
        # Prefer accurate TokenCounter
        from routers.utils.token_stats import TokenCounter, compute_prompt_tokens
        
        counter = TokenCounter()  # uses tiktoken cl100k_base
        
        N = len(meta)
        token_counts = np.zeros(N, dtype=int)
        
        for i in range(N):
            row = meta.iloc[i]
            
            # Build sample dict (compatible with compute_prompt_tokens input format)
            sample = {}
            
            # Extract prompt/question
            if 'question' in meta.columns:
                sample['prompt'] = str(row['question'])
            elif 'prompt' in meta.columns:
                sample['prompt'] = str(row['prompt'])
            else:
                sample['prompt'] = ''
            
            # Extract assets
            if 'assets' in meta.columns and row['assets'] is not None:
                sample['assets'] = row['assets']
            else:
                sample['assets'] = []
            
            # Compute tokens
            token_stats = compute_prompt_tokens(sample, counter)
            token_counts[i] = token_stats['total_input_tokens']
        
        # If no tokens were counted, use a default value
        token_counts[token_counts == 0] = 100
        
        return token_counts
        
    except ImportError:
        warnings.warn(
            "TokenCounter not available, falling back to simple estimation. "
            "Install tiktoken for accurate token counting: pip install tiktoken"
        )
        # Fall back to simple estimation
        return _simple_token_estimation(meta, tokenizer)


def _simple_token_estimation(meta, tokenizer=None) -> np.ndarray:
    """
    Simple token estimation (fallback).

    Strategy:
    - Text: whitespace tokenization × 1.3
    - Images: fixed 256 tokens (generic estimate)
    """
    N = len(meta)
    token_counts = np.zeros(N, dtype=int)
    
    # Text tokens
    text_col = None
    if 'question' in meta.columns:
        text_col = 'question'
    elif 'prompt' in meta.columns:
        text_col = 'prompt'
    
    if text_col is not None:
        for i, text in enumerate(meta[text_col]):
            if isinstance(text, str):
                if tokenizer is not None:
                    try:
                        tokens = tokenizer.encode(text, add_special_tokens=True)
                        token_counts[i] += len(tokens)
                    except:
                        # Fallback
                        words = text.split()
                        token_counts[i] += int(len(words) * 1.3)
                else:
                    words = text.split()
                    token_counts[i] += int(len(words) * 1.3)
    
    # Image tokens (generic estimate: 256 tokens per image)
    if 'assets' in meta.columns:
        for i, assets in enumerate(meta['assets']):
            if assets and len(assets) > 0:
                # Count images
                num_images = sum(1 for a in assets if a.get('type') in ['image', 'image_tsv', 'image_url'])
                token_counts[i] += num_images * 256  # 256 tokens per image
    
    # If no tokens were counted, use a default value
    token_counts[token_counts == 0] = 100
    
    return token_counts


class LatencyProfiler:
    """Latency profiler for measuring router inference time (per-token basis)"""
    
    def __init__(self, warmup_runs: int = 5, test_runs: int = 100, tokenizer=None):
        """
        Args:
            warmup_runs: Number of warmup iterations (to warm up GPU/cache)
            test_runs: Number of test iterations for timing
            tokenizer: Optional tokenizer for accurate token counting
        """
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
        self.tokenizer = tokenizer
    
    def profile_router(
        self,
        router,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        meta=None,
        batch_sizes: List[int] = [1, 8, 16, 32],
        device: Optional[str] = None
    ) -> Dict:
        """
        Profile router inference latency at different batch sizes
        
        Primary: per-token latency (ms/token, tokens/s)
        Secondary: per-sample latency (ms/sample, samples/s)
        
        Args:
            router: Router instance to profile
            X: Fused features (N x D), optional
            X_text: Text features (N x D_text), optional
            X_vision: Vision features (N x D_vision), optional
            meta: Metadata DataFrame, optional (used to count tokens)
            batch_sizes: List of batch sizes to test
            device: Device to use ('cuda', 'cpu', None=auto)
        
        Returns:
            Dict with latency statistics per batch size (token-based + sample-based)
        """
        # Determine device
        if device is None:
            if hasattr(router, 'device'):
                device = router.device
            else:
                device = 'cpu'
        
        # Synchronize if using CUDA
        def sync():
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
        
        results = {
            'device': device,
            'router_type': type(router).__name__,
            'batch_results': {}
        }
        
        # Get total available samples
        if X is not None:
            total_samples = len(X)
        elif X_text is not None:
            total_samples = len(X_text)
        elif X_vision is not None:
            total_samples = len(X_vision)
        elif meta is not None:
            total_samples = len(meta)
        else:
            raise ValueError("Must provide at least one of: X, X_text, X_vision, meta")
        
        # Count tokens (if meta is provided)
        token_counts = None
        token_method = "unknown"
        if meta is not None:
            try:
                # Detect token counting method
                try:
                    from routers.utils.token_stats import TokenCounter
                    token_method = "tiktoken (exact)"
                except ImportError:
                    token_method = "simple estimation"
                
                token_counts = estimate_tokens_from_meta(meta, self.tokenizer)
                total_tokens = token_counts.sum()
                avg_tokens_per_sample = token_counts.mean()
                results['total_tokens'] = int(total_tokens)
                results['avg_tokens_per_sample'] = float(avg_tokens_per_sample)
                results['token_counting_method'] = token_method
                
                print(
                    f"  Token stats ({token_method}): {total_samples} samples, {total_tokens} tokens "
                    f"(avg {avg_tokens_per_sample:.1f} tokens/sample)"
                )
            except Exception as e:
                warnings.warn(f"Failed to count tokens: {e}. Falling back to a fixed estimate.")
                token_counts = np.ones(total_samples, dtype=int) * 100  # default: 100 tokens/sample
                results['token_counting_method'] = "fixed_estimate"
        
        for batch_size in batch_sizes:
            if batch_size > total_samples:
                print(f"  Skipping batch_size={batch_size} (exceeds available {total_samples} samples)")
                continue
            
            # Extract batch
            batch_X = X[:batch_size] if X is not None else None
            batch_X_text = X_text[:batch_size] if X_text is not None else None
            batch_X_vision = X_vision[:batch_size] if X_vision is not None else None
            batch_meta = meta.iloc[:batch_size] if meta is not None else None
            
            # Warmup
            for _ in range(self.warmup_runs):
                try:
                    _ = router.predict(
                        X=batch_X,
                        X_text=batch_X_text,
                        X_vision=batch_X_vision,
                        meta=batch_meta
                    )
                except TypeError:
                    # Fallback for routers that don't support X_text/X_vision
                    if batch_meta is not None:
                        _ = router.predict(meta=batch_meta)
                    else:
                        raise ValueError("Router doesn't support provided input format")
                sync()
            
            # Timing runs
            times = []
            for _ in range(self.test_runs):
                sync()
                start_time = time.perf_counter()
                
                try:
                    _ = router.predict(
                        X=batch_X,
                        X_text=batch_X_text,
                        X_vision=batch_X_vision,
                        meta=batch_meta
                    )
                except TypeError:
                    # Fallback for routers that don't support X_text/X_vision
                    if batch_meta is not None:
                        _ = router.predict(meta=batch_meta)
                    else:
                        raise ValueError("Router doesn't support provided input format")
                
                sync()
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            times_ms = np.array(times) * 1000  # Convert to milliseconds
            per_sample_ms = times_ms / batch_size
            
            # Compute per-sample metrics
            batch_result = {
                'batch_size': batch_size,
                'total_time_ms': {
                    'mean': float(times_ms.mean()),
                    'std': float(times_ms.std()),
                    'min': float(times_ms.min()),
                    'max': float(times_ms.max()),
                    'p50': float(np.percentile(times_ms, 50)),
                    'p95': float(np.percentile(times_ms, 95)),
                    'p99': float(np.percentile(times_ms, 99))
                },
                'per_sample_ms': {
                    'mean': float(per_sample_ms.mean()),
                    'std': float(per_sample_ms.std()),
                    'min': float(per_sample_ms.min()),
                    'max': float(per_sample_ms.max()),
                    'p50': float(np.percentile(per_sample_ms, 50)),
                    'p95': float(np.percentile(per_sample_ms, 95)),
                    'p99': float(np.percentile(per_sample_ms, 99))
                },
                'throughput_samples_per_sec': float(batch_size / (times_ms.mean() / 1000))
            }
            
            # Compute per-token metrics (primary)
            if token_counts is not None:
                batch_token_count = token_counts[:batch_size].sum()
                per_token_ms = times_ms / batch_token_count
                
                batch_result['batch_token_count'] = int(batch_token_count)
                batch_result['per_token_ms'] = {
                    'mean': float(per_token_ms.mean()),
                    'std': float(per_token_ms.std()),
                    'min': float(per_token_ms.min()),
                    'max': float(per_token_ms.max()),
                    'p50': float(np.percentile(per_token_ms, 50)),
                    'p95': float(np.percentile(per_token_ms, 95)),
                    'p99': float(np.percentile(per_token_ms, 99))
                }
                batch_result['throughput_tokens_per_sec'] = float(batch_token_count / (times_ms.mean() / 1000))
            
            results['batch_results'][batch_size] = batch_result
            
            # Print results (prefer per-token metrics when available)
            if token_counts is not None and 'per_token_ms' in batch_result:
                print(f"  Batch size {batch_size:3d}: "
                      f"{batch_result['per_token_ms']['mean']:.4f} ± {batch_result['per_token_ms']['std']:.4f} ms/token "
                      f"(throughput: {batch_result['throughput_tokens_per_sec']:.1f} tokens/s, "
                      f"{batch_result['throughput_samples_per_sec']:.1f} samples/s)")
            else:
                print(f"  Batch size {batch_size:3d}: "
                      f"{per_sample_ms.mean():.3f} ± {per_sample_ms.std():.3f} ms/sample "
                      f"(throughput: {batch_result['throughput_samples_per_sec']:.1f} samples/s)")
        
        # Summary: use batch_size=1 as the primary latency metric
        if 1 in results['batch_results']:
            results['latency_ms_per_sample'] = results['batch_results'][1]['per_sample_ms']['mean']
            # Per-token metrics (primary)
            if 'per_token_ms' in results['batch_results'][1]:
                results['latency_ms_per_token'] = results['batch_results'][1]['per_token_ms']['mean']
                results['throughput_tokens_per_sec'] = results['batch_results'][1]['throughput_tokens_per_sec']
        elif batch_sizes:
            # Use the smallest tested batch size
            smallest_batch = min(results['batch_results'].keys())
            results['latency_ms_per_sample'] = results['batch_results'][smallest_batch]['per_sample_ms']['mean']
            if 'per_token_ms' in results['batch_results'][smallest_batch]:
                results['latency_ms_per_token'] = results['batch_results'][smallest_batch]['per_token_ms']['mean']
                results['throughput_tokens_per_sec'] = results['batch_results'][smallest_batch]['throughput_tokens_per_sec']
        
        return results
    
    def profile_multiple_routers(
        self,
        routers: Dict[str, object],
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        meta=None,
        batch_sizes: List[int] = [1, 8, 16, 32]
    ) -> Dict:
        """
        Profile multiple routers
        
        Args:
            routers: Dict of {router_name: router_instance}
            Other args same as profile_router
        
        Returns:
            Dict of {router_name: latency_results}
        """
        all_results = {}
        
        for router_name, router in routers.items():
            print(f"\n📊 Profiling {router_name}...")
            try:
                results = self.profile_router(
                    router, X, X_text, X_vision, meta, batch_sizes
                )
                all_results[router_name] = results
                
                # Prefer per-token metrics
                if 'latency_ms_per_token' in results:
                    print(f"  ✓ Average latency: {results['latency_ms_per_token']:.4f} ms/token "
                          f"({results['throughput_tokens_per_sec']:.1f} tokens/s)")
                elif 'latency_ms_per_sample' in results:
                    print(f"  ✓ Average latency: {results['latency_ms_per_sample']:.3f} ms/sample")
            except Exception as e:
                print(f"  ⚠️  Failed to profile {router_name}: {e}")
                all_results[router_name] = {'error': str(e)}
        
        return all_results


def format_latency_report(results: Dict, output_file: Optional[str] = None) -> str:
    """
    Format latency profiling results into a readable report
    
    Prefer per-token metrics (primary) and also show per-sample metrics (secondary).
    
    Args:
        results: Results from LatencyProfiler.profile_multiple_routers
        output_file: Optional file path to save the report
    
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Router Latency Profiling Report")
    lines.append("=" * 80)
    lines.append("")
    
    # Per-Token Summary (primary)
    has_token_metrics = any('latency_ms_per_token' in r for r in results.values() if 'error' not in r)
    
    if has_token_metrics:
        lines.append("📊 Per-Token Latency Summary (primary, @ batch_size=1):")
        lines.append("-" * 80)
        lines.append(f"{'Router':<30} {'Device':<10} {'ms/token':<15} {'Throughput (tok/s)':<20}")
        lines.append("-" * 80)
        
        for router_name, router_result in results.items():
            if 'error' in router_result:
                lines.append(f"{router_name:<30} {'N/A':<10} {'ERROR':<15} {router_result['error']}")
                continue
            
            device = router_result.get('device', 'N/A')
            
            if 'latency_ms_per_token' in router_result:
                latency_token = router_result['latency_ms_per_token']
                throughput_token = router_result.get('throughput_tokens_per_sec', 'N/A')
                if isinstance(throughput_token, (int, float)):
                    lines.append(f"{router_name:<30} {device:<10} {latency_token:<15.4f} {throughput_token:<20.1f}")
                else:
                    lines.append(f"{router_name:<30} {device:<10} {latency_token:<15.4f} {throughput_token:<20}")
            else:
                lines.append(f"{router_name:<30} {device:<10} {'N/A':<15} {'N/A':<20}")
        
        lines.append("-" * 80)
        lines.append("")
    
    # Per-Sample Summary (secondary)
    lines.append("📈 Per-Sample Latency Summary (secondary, @ batch_size=1):")
    lines.append("-" * 80)
    lines.append(f"{'Router':<30} {'Device':<10} {'ms/sample':<15} {'Throughput (samp/s)':<20}")
    lines.append("-" * 80)
    
    for router_name, router_result in results.items():
        if 'error' in router_result:
            continue
        
        device = router_result.get('device', 'N/A')
        latency = router_result.get('latency_ms_per_sample', 'N/A')
        
        if 1 in router_result.get('batch_results', {}):
            throughput = router_result['batch_results'][1]['throughput_samples_per_sec']
            if isinstance(latency, (int, float)):
                lines.append(f"{router_name:<30} {device:<10} {latency:<15.3f} {throughput:<20.1f}")
            else:
                lines.append(f"{router_name:<30} {device:<10} {latency:<15} {throughput:<20.1f}")
        else:
            lines.append(f"{router_name:<30} {device:<10} {latency:<15} {'N/A':<20}")
    
    lines.append("-" * 80)
    lines.append("")
    
    # Detailed results per router
    for router_name, router_result in results.items():
        if 'error' in router_result:
            continue
        
        lines.append(f"\n{router_name}:")
        lines.append(f"  Device: {router_result.get('device', 'N/A')}")
        lines.append(f"  Router Type: {router_result.get('router_type', 'N/A')}")
        
        # Token statistics
        if 'total_tokens' in router_result:
            token_method = router_result.get('token_counting_method', 'unknown')
            lines.append(f"  Token Counting Method: {token_method}")
            lines.append(f"  Total Tokens: {router_result['total_tokens']:,}")
            lines.append(f"  Avg Tokens/Sample: {router_result['avg_tokens_per_sample']:.1f}")
        
        lines.append("  Batch Size Results:")
        
        for batch_size, batch_result in router_result.get('batch_results', {}).items():
            per_sample = batch_result['per_sample_ms']
            
            # Prefer per-token metrics
            if 'per_token_ms' in batch_result:
                per_token = batch_result['per_token_ms']
                lines.append(f"    Batch {batch_size:3d}: "
                            f"{per_token['mean']:.4f} ± {per_token['std']:.4f} ms/token "
                            f"(p50: {per_token['p50']:.4f}, p95: {per_token['p95']:.4f}, "
                            f"p99: {per_token['p99']:.4f})")
                lines.append(f"             "
                            f"{per_sample['mean']:.3f} ± {per_sample['std']:.3f} ms/sample "
                            f"({batch_result.get('batch_token_count', 'N/A')} tokens in batch)")
            else:
                lines.append(f"    Batch {batch_size:3d}: "
                            f"{per_sample['mean']:.3f} ± {per_sample['std']:.3f} ms/sample "
                            f"(p50: {per_sample['p50']:.3f}, p95: {per_sample['p95']:.3f}, "
                            f"p99: {per_sample['p99']:.3f})")
    
    lines.append("")
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"  Latency report saved to: {output_file}")
    
    return report


if __name__ == '__main__':
    # Example usage
    print("Latency profiler module loaded.")
    print("Use: from routers.utils.latency_profiler import LatencyProfiler")

