#!/usr/bin/env python3
"""
Token statistics utilities (batch-optimized version)

Optimization strategy:
1. Batch-read all VLMEvalKit result files into one large DataFrame
2. Compute output tokens with vectorized operations
3. Batch merge + compute costs to avoid per-sample loops

Expected speedup:
- 15 models × 14 datasets × 30k samples ≈ ~6M records
- Original: per-sample processing, ~30–60 minutes
- Optimized: batch processing, ~5–10 minutes
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("⚠️  Warning: tiktoken not installed. Please install: pip install tiktoken")


class BatchTokenCounter:
    """Batch token counter (vectorized-friendly)."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                print(f"⚠️  Warning: Cannot load tiktoken encoding {encoding_name}: {e}")
                self.encoder = None
        else:
            self.encoder = None
    
    def count_text_tokens(self, text: str) -> int:
        """Count tokens for a single text."""
        if not text or pd.isna(text):
            return 0
        
        if self.encoder is not None:
            try:
                return len(self.encoder.encode(str(text)))
            except:
                pass
        
        # Fallback
        words = len(str(text).split())
        return int(words * 1.33)
    
    def count_text_tokens_batch(self, texts: pd.Series) -> pd.Series:
        """Batch-count text tokens."""
        if self.encoder is not None:
            try:
                # Use apply; internal caching may help
                return texts.apply(lambda x: self.count_text_tokens(x))
            except:
                pass
        
        # Fallback: vectorized word-count heuristic
        word_counts = texts.fillna('').astype(str).str.split().str.len()
        return (word_counts * 1.33).astype(int)
    
    def estimate_image_tokens(self, num_images: int, model_type: str = "generic") -> int:
        """Estimate tokens for images."""
        if num_images == 0:
            return 0
        return num_images * 256  # default: 256 tokens per image


def load_all_vlmevalkit_results_batch(
    vlmevalkit_dir: Path,
    models: List[str],
    datasets: List[str]
) -> pd.DataFrame:
    """
    Batch-read all VLMEvalKit result files.
    
    Returns:
        DataFrame with columns: [sample_id, model, dataset, prediction, is_correct]
    """
    print("\n📥 Batch-loading VLMEvalKit results...")
    
    all_results = []
    file_patterns = [
        '*_openai_result.xlsx',  # MCQ type
        '*_gpt-4o-mini.xlsx',     # Math type
        '*_auxmatch.xlsx',        # Y/N type
        '*_VAL.xlsx',             # VQA type (TextVQA_VAL, DocVQA_VAL, InfoVQA_VAL)
        '*_TEST.xlsx',            # VQA type (ChartQA_TEST, AI2D_TEST)
        '*_MINI.xlsx',            # Math type (MathVerse_MINI, MathVista_MINI, MathVision_MINI)
        '*OCRBench.xlsx',         # OCR type
        # Inference files (often {model}_{dataset}.xlsx); some datasets don't end with _VAL/_TEST/_MINI
        '*MMBench_DEV_EN_V11.xlsx',
        '*MMStar.xlsx',
        '*RealWorldQA.xlsx',
        '*HallusionBench.xlsx',
    ]
    
    # Extend model list (include newly added models)
    all_models = models + [
        'Gemma3-27B', 'Janus-Pro-1B', 'Janus-Pro-7B',
        'Kimi-VL-A3B-Thinking-2506', 'Pixtral-12B', 'Qianfan-VL-8B',
        'deepseek_vl2', 'deepseek_vl2_tiny'
    ]
    all_models = list(set(all_models))  # deduplicate
    
    # Collect all matching files
    result_files = []
    exclude_patterns = ['_results.xlsx', '_extract.xlsx', '_score.xlsx', '_acc.xlsx']
    
    for model in all_models:
        model_dir = vlmevalkit_dir / model
        if not model_dir.exists():
            continue
        
        for pattern in file_patterns:
            for fpath in model_dir.glob(pattern):
                # Exclude derived files
                if any(excl in str(fpath) for excl in exclude_patterns):
                    continue
                result_files.append(fpath)
    
    print(f"  Found {len(result_files)} result files")
    
    # Batch-read all files
    for fpath in tqdm(result_files, desc="Reading result files"):
        try:
            df = pd.read_excel(fpath, engine='openpyxl')
            
            # Extract model and dataset names
            model_name = fpath.parent.name
            if model_name not in all_models:
                continue
            
            # Extract dataset name from filename
            fname = fpath.stem
            for suffix in ['_openai_result', '_gpt-4o-mini', '_auxmatch', '_result']:
                fname = fname.replace(suffix, '')
            if fname.startswith(model_name + '_'):
                dataset_name = fname[len(model_name)+1:]
            else:
                dataset_name = fname
            
            # Normalize column names
            if 'id' in df.columns:
                id_col = 'id'
            elif 'index' in df.columns:
                id_col = 'index'
            else:
                continue
            
            # Find prediction column
            prediction_col = None
            for col in ['prediction', 'answer', 'response', 'output']:
                if col in df.columns:
                    prediction_col = col
                    break
            
            if prediction_col is None:
                continue
            
            # Normalize dataset name
            dataset_normalized = normalize_dataset_name(dataset_name)
            
            # Build sample_id
            df['sample_id'] = dataset_normalized + '/' + df[id_col].astype(str)
            df['model'] = model_name
            df['dataset'] = dataset_normalized
            df['prediction'] = df[prediction_col].fillna('').astype(str)
            
            # Extract is_correct (if present)
            if 'is_correct' in df.columns:
                df['is_correct'] = df['is_correct']
            elif 'score' in df.columns:
                df['is_correct'] = df['score']
            elif 'hit' in df.columns:
                df['is_correct'] = df['hit']
            else:
                df['is_correct'] = np.nan
            
            # Keep only required columns
            result_df = df[['sample_id', 'model', 'dataset', 'prediction', 'is_correct']].copy()
            all_results.append(result_df)
            
        except Exception as e:
            continue
    
    if not all_results:
        print("  ⚠️  No results found")
        return pd.DataFrame()
    
    # Merge all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Deduplicate (same sample_id×model may appear in multiple files)
    combined_df = combined_df.drop_duplicates(subset=['sample_id', 'model'], keep='first')
    
    print(f"  ✓ Loaded {len(combined_df)} records")
    print(f"  ✓ Covered {len(combined_df['sample_id'].unique())} samples")
    print(f"  ✓ Covered {len(combined_df['model'].unique())} models")
    
    # Coverage by model
    model_counts = combined_df['model'].value_counts()
    print("\n  📊 Records by model:")
    for model, count in model_counts.items():
        print(f"    - {model}: {count:,}")
    
    return combined_df


def normalize_dataset_name(dataset_name: str) -> str:
    """Normalize dataset name."""
    mappings = {
        'MMMU_DEV_VAL': ['MMMU_DEV_VAL', 'mmmu_dev_val', 'MMMU-DEV-VAL', 'MMMU_DEV'],
        'MathVista_MINI': ['MathVista_MINI', 'mathvista_mini', 'MathVista-MINI'],
        'MathVision_MINI': ['MathVision_MINI', 'mathvision_mini', 'MathVision-MINI'],
        'MathVerse_MINI': ['MathVerse_MINI', 'mathverse_mini', 'MathVerse-MINI'],
        'MMBench_DEV_EN_V11': ['MMBench_DEV_EN_V11', 'mmbench_dev_en_v11', 'MMBench-DEV-EN-V11'],
        'RealWorldQA': ['RealWorldQA', 'realworldqa', 'RealWorld-QA'],
        'MMStar': ['MMStar', 'mmstar', 'MM-Star'],
        'HallusionBench': ['HallusionBench', 'hallusionbench', 'Hallusion-Bench'],
        'TextVQA_VAL': ['TextVQA_VAL', 'textvqa_val', 'TextVQA-VAL'],
        'ChartQA_TEST': ['ChartQA_TEST', 'chartqa_test', 'ChartQA-TEST'],
        'DocVQA_VAL': ['DocVQA_VAL', 'docvqa_val', 'DocVQA-VAL'],
        'InfoVQA_VAL': ['InfoVQA_VAL', 'infovqa_val', 'InfoVQA-VAL'],
        'AI2D_TEST': ['AI2D_TEST', 'ai2d_test', 'AI2D-TEST'],
        'OCRBench': ['OCRBench', 'ocrbench', 'OCR-Bench'],
    }
    
    for standard, variants in mappings.items():
        if dataset_name in variants:
            return standard
    
    return dataset_name.upper()


def compute_dataset_token_stats_batch(
    benchmark_dir: Path,
    vlmevalkit_dir: Optional[Path],
    models: List[str],
    datasets: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Batch-compute token statistics (optimized).
    
    Returns:
        Tuple of (input_token_df, output_token_df)
    """
    counter = BatchTokenCounter()
    
    # 1. Read all benchmark samples (batch)
    print("\n📊 Batch-reading benchmark samples...")
    all_samples = []
    
    samples_files = []
    for task_dir in benchmark_dir.iterdir():
        if not task_dir.is_dir():
            continue
        for samples_file in task_dir.glob("*_samples.jsonl"):
            dataset_name = samples_file.stem.replace('_samples', '').upper()
            if datasets is None or dataset_name in datasets:
                samples_files.append((samples_file, dataset_name))
    
    for samples_file, dataset_name in tqdm(samples_files, desc="Reading samples"):
        with open(samples_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                all_samples.append({
                    'sample_id': sample['sample_id'],
                    'dataset': dataset_name,
                    'task_type': sample.get('task_type', 'unknown'),
                    'prompt': sample.get('prompt', ''),
                    'num_images': len([a for a in sample.get('assets', []) 
                                     if a.get('type') in ['image', 'image_tsv', 'image_url']])
                })
    
    samples_df = pd.DataFrame(all_samples)
    print(f"  ✓ Read {len(samples_df)} samples")
    
    # 2. Batch-compute input tokens (vectorized)
    print("\n📝 Batch-computing input tokens...")
    samples_df['text_tokens'] = counter.count_text_tokens_batch(samples_df['prompt'])
    samples_df['image_tokens'] = samples_df['num_images'] * 256  # vectorized
    samples_df['total_input_tokens'] = samples_df['text_tokens'] + samples_df['image_tokens']
    
    input_df = samples_df[['sample_id', 'dataset', 'task_type', 
                           'text_tokens', 'image_tokens', 'total_input_tokens', 'num_images']].copy()
    
    print("  ✓ Done")
    
    # 3. Batch-read VLMEvalKit results
    if vlmevalkit_dir and vlmevalkit_dir.exists():
        results_df = load_all_vlmevalkit_results_batch(vlmevalkit_dir, models, 
                                                        datasets if datasets else [])

        # Keep only samples in this benchmark (inference dirs may contain a broader set)
        if len(results_df) > 0:
            valid_sample_ids = set(samples_df['sample_id'].tolist())
            before = len(results_df)
            results_df = results_df[results_df['sample_id'].isin(valid_sample_ids)].copy()
            after = len(results_df)
            dropped = before - after
            if dropped > 0:
                print(f"  ⚠️  Dropped {dropped:,} inference records not in BENCHMARKS (inference directory may contain a broader set)")
        
        # 4. Batch-compute output tokens (vectorized)
        if len(results_df) > 0:
            print("\n📝 Batch-computing output tokens...")
            results_df['output_tokens'] = counter.count_text_tokens_batch(results_df['prediction'])
            results_df['is_actual'] = True
            
            output_df = results_df[['sample_id', 'model', 'output_tokens', 'is_actual']].copy()
            
            # Coverage stats
            actual_count = len(output_df)
            total_expected = len(samples_df) * len(models)
            print(f"  ✓ Actual predictions: {actual_count:,}/{total_expected:,} ({100*actual_count/total_expected:.1f}%)")
            
            # 5. Fill missing sample×model with estimates
            print("\n📝 Filling missing records with estimates...")
            # Vectorized cartesian product (sample_id × model) to avoid nested Python loops
            all_comb_df = (
                pd.MultiIndex.from_product([samples_df['sample_id'].tolist(), models], names=['sample_id', 'model'])
                .to_frame(index=False)
                .merge(samples_df[['sample_id', 'task_type']], on='sample_id', how='left')
            )
            
            # Merge with actual output tokens
            output_df_full = all_comb_df.merge(
                output_df,
                on=['sample_id', 'model'],
                how='left'
            )
            
            # Fill missing entries with estimates
            task_estimates = {
                'vqa_mc': 10,
                'classification': 10,
                'vqa_oe': 50,
                'ocr_qa': 50,
                'doc_qa': 150,
                'math_logic': 150,
                'math': 150,
                'unknown': 100
            }
            
            def estimate_output_tokens(row):
                if pd.notna(row['output_tokens']):
                    return row['output_tokens']
                return task_estimates.get(row['task_type'], 100)
            
            output_df_full['output_tokens'] = output_df_full.apply(estimate_output_tokens, axis=1)
            output_df_full['is_actual'] = output_df_full['is_actual'].fillna(False)
            
            output_df_final = output_df_full[['sample_id', 'model', 'output_tokens', 'is_actual']].copy()
            
            missing_count = (~output_df_final['is_actual']).sum()
            print(f"  ✓ Filled {missing_count:,} estimated values")
            
            return input_df, output_df_final
        else:
            print("\n  ⚠️  VLMEvalKit results not found; using estimates for all outputs")
    
    # Fallback: use estimates for all outputs
    print("\n📝 Generating output-token estimates...")
    output_records = []
    for _, sample_row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="Generating estimates"):
        task_type = sample_row['task_type']
        if task_type in ['vqa_mc', 'classification']:
            estimated_tokens = 10
        elif task_type in ['vqa_oe', 'ocr_qa']:
            estimated_tokens = 50
        elif task_type in ['doc_qa', 'math_logic', 'math']:
            estimated_tokens = 150
        else:
            estimated_tokens = 100
        
        for model in models:
            output_records.append({
                'sample_id': sample_row['sample_id'],
                'model': model,
                'output_tokens': estimated_tokens,
                'is_actual': False
            })
    
    output_df = pd.DataFrame(output_records)
    
    return input_df, output_df


def compute_token_based_costs_batch(
    input_tokens_df: pd.DataFrame,
    output_tokens_df: pd.DataFrame,
    pricing_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Batch-compute token-based costs (vectorized).
    
    Args:
        input_tokens_df: Input token statistics
        output_tokens_df: Output token statistics
        pricing_config: Pricing configuration
        
    Returns:
        Cost DataFrame
    """
    print("\n💰 Batch-computing token-based costs...")
    
    # Default pricing
    if pricing_config is None:
        pricing_config = {
            'InternVL2_5-78B': {'input': 5.0, 'output': 15.0},
            'Qwen2.5-VL-72B-Instruct': {'input': 4.0, 'output': 12.0},
            'Qwen2.5-VL-32B-Instruct': {'input': 2.0, 'output': 6.0},
            'MiMo-VL-7B-RL': {'input': 0.5, 'output': 1.5},
            'Phi-3.5-Vision': {'input': 0.3, 'output': 1.0},
            'SmolVLM2': {'input': 0.2, 'output': 0.6},
            'llava_next_vicuna_7b': {'input': 0.3, 'output': 1.0},
            # Newly added models
            'Gemma3-27B': {'input': 0.8, 'output': 2.4},
            'Janus-Pro-1B': {'input': 0.1, 'output': 0.3},
            'Janus-Pro-7B': {'input': 0.3, 'output': 0.9},
            'Kimi-VL-A3B-Thinking-2506': {'input': 0.15, 'output': 0.45},
            'Pixtral-12B': {'input': 0.4, 'output': 1.2},
            'Qianfan-VL-8B': {'input': 0.3, 'output': 0.9},
            'deepseek_vl2': {'input': 0.4, 'output': 1.2},
            'deepseek_vl2_tiny': {'input': 0.1, 'output': 0.3},
        }
    
    # Merge input and output tokens
    merged = output_tokens_df.merge(
        input_tokens_df[['sample_id', 'total_input_tokens']],
        on='sample_id',
        how='left'
    )
    
    # Create a price mapping DataFrame (avoid row-wise apply)
    price_df = pd.DataFrame([
        {'model': k, 'input_price': v['input'], 'output_price': v['output']}
        for k, v in pricing_config.items()
    ])
    
    # Merge prices
    merged = merged.merge(price_df, on='model', how='left')
    
    # Fill missing prices (use defaults)
    merged['input_price'] = merged['input_price'].fillna(1.0)
    merged['output_price'] = merged['output_price'].fillna(3.0)
    
    # Vectorized cost computation
    merged['input_cost'] = (merged['total_input_tokens'] / 1_000_000) * merged['input_price']
    merged['output_cost'] = (merged['output_tokens'] / 1_000_000) * merged['output_price']
    merged['total_cost'] = merged['input_cost'] + merged['output_cost']
    
    result_df = merged[['sample_id', 'model', 'total_input_tokens', 'output_tokens',
                        'input_cost', 'output_cost', 'total_cost']].copy()
    
    print(f"  ✓ Computed {len(result_df):,} cost records")
    print(f"  ✓ Total cost: ${result_df['total_cost'].sum():.6f}")
    
    return result_df

