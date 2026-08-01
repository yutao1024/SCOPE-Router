#!/usr/bin/env python3
"""
Token statistics utilities (accurate version)

Functions:
1. Use a real tokenizer (tiktoken) to accurately count text prompt tokens
2. Estimate image tokens per sample (based on model type)
3. Extract model outputs from VLMEvalKit result files and count output tokens

Usage:
    from routers.utils.token_stats import TokenCounter
    
    counter = TokenCounter()
    text_tokens = counter.count_text_tokens(prompt)
    output_tokens = counter.count_text_tokens(model_output)
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
    print("   Falling back to simple word-based estimation")


class TokenCounter:
    """Token counter (using a real tokenizer)."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize Token Counter
        
        Args:
            encoding_name: tiktoken encoding name (default cl100k_base for GPT-4/GPT-3.5-turbo)
        """
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
        """
        Calculate text tokens using real tokenizer
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        if self.encoder is not None:
            try:
                return len(self.encoder.encode(str(text)))
            except Exception as e:
                print(f"⚠️  Token encoding error: {e}, using simple estimation")
        
        # Fallback to simple word-based estimation
        words = len(str(text).split())
        return int(words * 1.33)  # 1 word ≈ 1.33 tokens
    
    def estimate_image_tokens(self, 
                            image_size: Optional[Tuple[int, int]] = None,
                            model_type: str = "generic") -> int:
        """
        Estimate image tokens
        
        Different models process images differently:
        - CLIP/OpenAI: Image split into patches, each patch = 1 token
        - LLaVA: Grid patches, quantity depends on resolution
        - Qwen-VL: Adaptive patch quantity
        
        Args:
            image_size: Image size (width, height), if None use default estimate
            model_type: Model type ("generic", "clip", "llava", "qwen", etc)
            
        Returns:
            Estimated token count
        """
        if image_size is None:
            # Default estimate (based on common 336x336 or 448x448 input)
            if model_type == "clip":
                return 256  # CLIP ViT-L/14: 16x16 patches
            elif model_type == "llava":
                return 576  # LLaVA: 24x24 grid
            elif model_type == "qwen":
                return 256  # Qwen-VL: adaptive
            else:
                return 256  # Generic estimate
        
        width, height = image_size
        
        # Calculate patch count based on model type
        if model_type == "clip":
            patch_size = 14
            num_patches_w = width // patch_size
            num_patches_h = height // patch_size
            return num_patches_w * num_patches_h + 1  # +1 for CLS token
        
        elif model_type == "llava":
            target_size = 336
            grid_size = 24
            scale = min(target_size / width, target_size / height)
            scaled_w = int(width * scale)
            scaled_h = int(height * scale)
            return (scaled_w // grid_size) * (scaled_h // grid_size)
        
        elif model_type == "qwen":
            area = width * height
            if area <= 224 * 224:
                return 256
            elif area <= 448 * 448:
                return 512
            else:
                return 1024
        
        else:
            # Generic estimate: assume 14x14 patch size
            patch_size = 14
            num_patches = (width // patch_size) * (height // patch_size)
            return max(256, min(num_patches, 1024))  # Limit between 256-1024


def extract_model_dataset_from_path(filepath: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract model name and dataset name from VLMEvalKit result file path
    Reference: build_benchmark_from_vlmevalkit.py
    """
    basename = filepath.stem
    
    # Remove common suffixes
    name = basename.replace('_openai_result', '') \
                   .replace('_result', '') \
                   .replace('_gpt-4o-mini', '') \
                   .replace('_gpt-4o-mini_score', '') \
                   .replace('_gpt-4o-mini_extract', '') \
                   .replace('_auxmatch', '') \
                   .replace('_results', '')
    
    # Extract model from path
    path_parts = filepath.parts
    model = None
    
    target_models = [
        'InternVL2_5-78B', 'MiMo-VL-7B-RL', 'Phi-3.5-Vision',
        'Qwen2.5-VL-32B-Instruct', 'Qwen2.5-VL-72B-Instruct',
        'SmolVLM2', 'llava_next_vicuna_7b'
    ]
    
    for part in path_parts:
        if part in target_models:
            model = part
            break
    
    if model is None:
        return None, None
    
    # Extract dataset from filename
    if name.startswith(model + '_'):
        dataset = name[len(model)+1:]
    else:
        dataset = name
    
    return model, dataset


def normalize_dataset_name(dataset_name: str) -> str:
    """
    Normalize dataset name to match benchmark format
    """
    # Common mappings
    mappings = {
        'MMMU_DEV_VAL': ['MMMU_DEV_VAL', 'mmmu_dev_val', 'MMMU-DEV-VAL'],
        'MMMU_DEV': ['MMMU_DEV', 'mmmu_dev', 'MMMU-DEV'],
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
        'OCRBench': ['OCRBench', 'ocrbench', 'OCR-Bench']
    }
    
    dataset_upper = dataset_name.upper().replace('-', '_')
    
    for canonical, variants in mappings.items():
        for variant in variants:
            if variant.upper().replace('-', '_') == dataset_upper:
                return canonical
    
    return dataset_name


def load_vlmevalkit_predictions(
    vlmevalkit_dir: Path,
    target_models: List[str],
    target_datasets: List[str]
) -> Dict[str, Dict[str, str]]:
    """
    Load model predictions from VLMEvalKit result files
    Uses robust extraction logic from build_benchmark_from_vlmevalkit.py
    
    Args:
        vlmevalkit_dir: VLMEvalKit output directory
        target_models: List of model names
        target_datasets: List of dataset names
        
    Returns:
        Dictionary: {sample_id: {model_name: prediction_text}}
    """
    print("\n📥 Loading model predictions from VLMEvalKit...")
    
    predictions = {}  # {sample_id: {model: prediction}}
    vlmevalkit_dir = Path(vlmevalkit_dir)
    
    # Normalize target dataset names
    normalized_datasets = [normalize_dataset_name(ds) for ds in target_datasets]
    
    # Find all xlsx files
    all_xlsx_files = list(vlmevalkit_dir.rglob("*.xlsx"))
    
    # Filter by file types (exclude intermediate scoring files)
    valid_files = []
    for f in all_xlsx_files:
        basename = f.name
        # Skip intermediate files (but keep _openai_result.xlsx and _result.xlsx)
        if any(x in basename for x in ['_acc.xlsx', '_extract.xlsx', '_score.xlsx']):
            continue
        valid_files.append(f)
    
    print(f"  Found {len(valid_files)} result files")
    
    processed_count = 0
    for result_file in tqdm(valid_files, desc="Loading predictions"):
        try:
            # Extract model and dataset from path
            model, dataset = extract_model_dataset_from_path(result_file)
            
            if model is None or model not in target_models:
                continue
            
            # Normalize dataset name
            dataset_normalized = normalize_dataset_name(dataset)
            
            if dataset_normalized not in normalized_datasets:
                continue
            
            # Read Excel file
            df = pd.read_excel(result_file, engine='openpyxl')
            
            # Find prediction column
            prediction_col = None
            for col in ['prediction', 'Prediction', 'pred', 'answer']:
                if col in df.columns:
                    prediction_col = col
                    break
            
            if prediction_col is None:
                continue
            
            # Find sample ID column
            # Priority: id > sample_id > index (id column usually contains meaningful string IDs)
            if 'id' in df.columns:
                sample_id_col = 'id'
            elif 'sample_id' in df.columns:
                sample_id_col = 'sample_id'
            elif 'index' in df.columns:
                sample_id_col = 'index'
            else:
                # Use DataFrame index
                df = df.reset_index()
                sample_id_col = 'index'
            
            # Extract predictions
            for _, row in df.iterrows():
                raw_id = str(row[sample_id_col])
                
                # Build sample_id in benchmark format: DATASET/id
                sample_id = f"{dataset_normalized}/{raw_id}"
                
                prediction = str(row.get(prediction_col, ''))
                
                if not prediction or prediction == 'nan':
                    continue
                
                if sample_id not in predictions:
                    predictions[sample_id] = {}
                
                predictions[sample_id][model] = prediction
            
            processed_count += 1
            
        except Exception as e:
            continue
    
    print(f"  ✓ Processed {processed_count} files")
    print(f"  ✓ Loaded predictions for {len(predictions)} samples")
    
    # Report coverage by model
    model_counts = {}
    for sid, model_preds in predictions.items():
        for model in model_preds:
            model_counts[model] = model_counts.get(model, 0) + 1
    
    print(f"  📊 Coverage by model:")
    for model in sorted(model_counts.keys()):
        print(f"    - {model}: {model_counts[model]} samples")
    
    return predictions


def compute_prompt_tokens(sample: Dict, counter: TokenCounter) -> Dict[str, int]:
    """
    Calculate prompt tokens for a sample (text + image)
    
    Args:
        sample: Sample dictionary containing prompt and assets
        counter: TokenCounter instance
        
    Returns:
        Token statistics: {'text_tokens': int, 'image_tokens': int, 'total_input_tokens': int}
    """
    # Calculate text tokens
    prompt = sample.get('prompt', '')
    text_tokens = counter.count_text_tokens(prompt)
    
    # Calculate image tokens
    assets = sample.get('assets', [])
    image_tokens = 0
    
    for asset in assets:
        if asset.get('type') in ['image', 'image_tsv', 'image_url']:
            # Estimate image tokens (use generic estimate)
            image_tokens += counter.estimate_image_tokens(model_type="generic")
    
    return {
        'text_tokens': text_tokens,
        'image_tokens': image_tokens,
        'total_input_tokens': text_tokens + image_tokens
    }


def compute_dataset_token_stats(
    benchmark_dir: Path,
    vlmevalkit_dir: Optional[Path],
    models: List[str],
    datasets: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate token statistics for entire benchmark
    
    Args:
        benchmark_dir: BENCHMARKS directory path
        vlmevalkit_dir: VLMEvalKit output directory (for extracting model outputs)
        models: List of model names
        datasets: List of datasets to process, if None process all
        
    Returns:
        Tuple of (input_token_df, output_token_df)
    """
    counter = TokenCounter()
    input_results = []
    output_results = []
    
    # Load model predictions from VLMEvalKit
    if vlmevalkit_dir and vlmevalkit_dir.exists():
        # Get list of datasets
        if datasets is None:
            datasets = []
            for task_dir in benchmark_dir.iterdir():
                if task_dir.is_dir():
                    for samples_file in task_dir.glob("*_samples.jsonl"):
                        dataset_name = samples_file.stem.replace('_samples', '').upper()
                        datasets.append(dataset_name)
        
        predictions = load_vlmevalkit_predictions(vlmevalkit_dir, models, datasets)
    else:
        print("\n⚠️  VLMEvalKit directory not provided, will use estimation for output tokens")
        predictions = {}
    
    # Count total samples for progress bar
    total_samples = 0
    samples_files = []
    for task_dir in benchmark_dir.iterdir():
        if not task_dir.is_dir():
            continue
        for samples_file in task_dir.glob("*_samples.jsonl"):
            dataset_name = samples_file.stem.replace('_samples', '').upper()
            if datasets is None or dataset_name in datasets:
                with open(samples_file, 'r') as f:
                    sample_count = sum(1 for _ in f)
                total_samples += sample_count
                samples_files.append((samples_file, dataset_name, sample_count))
    
    print(f"\n📊 Processing {total_samples} samples from {len(samples_files)} datasets...")
    
    # Process samples with progress bar
    with tqdm(total=total_samples, desc="Calculating tokens") as pbar:
        for samples_file, dataset_name, _ in samples_files:
            # Read samples
            with open(samples_file, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line)
                    sample_id = sample['sample_id']
                    
                    # Calculate input token statistics
                    token_stats = compute_prompt_tokens(sample, counter)
                    
                    input_results.append({
                        'sample_id': sample_id,
                        'dataset': dataset_name,
                        'task_type': sample.get('task_type', 'unknown'),
                        'text_tokens': token_stats['text_tokens'],
                        'image_tokens': token_stats['image_tokens'],
                        'total_input_tokens': token_stats['total_input_tokens'],
                        'num_images': len([a for a in sample.get('assets', []) 
                                         if a.get('type') in ['image', 'image_tsv', 'image_url']])
                    })
                    
                    # Calculate output tokens for each model
                    for model in models:
                        if sample_id in predictions and model in predictions[sample_id]:
                            # Use actual prediction
                            prediction_text = predictions[sample_id][model]
                            output_tokens = counter.count_text_tokens(prediction_text)
                        else:
                            # Fallback to task-based estimation
                            task_type = sample.get('task_type', 'unknown')
                            if task_type in ['vqa_mc', 'classification']:
                                output_tokens = 10  # Multiple choice usually very short
                            elif task_type in ['vqa_oe', 'ocr_qa']:
                                output_tokens = 50  # Open-ended Q&A medium length
                            elif task_type in ['doc_qa', 'math_logic', 'math']:
                                output_tokens = 150  # Document Q&A/math reasoning longer
                            else:
                                output_tokens = 100
                        
                        output_results.append({
                            'sample_id': sample_id,
                            'model': model,
                            'output_tokens': output_tokens,
                            'is_actual': (sample_id in predictions and model in predictions[sample_id])
                        })
                    
                    pbar.update(1)
    
    input_df = pd.DataFrame(input_results)
    output_df = pd.DataFrame(output_results)
    
    # Report statistics
    if len(output_df) > 0:
        actual_count = output_df['is_actual'].sum()
        total_count = len(output_df)
        print(f"\n  ✓ Actual predictions: {actual_count}/{total_count} ({100*actual_count/total_count:.1f}%)")
    
    return input_df, output_df


def aggregate_token_stats(
    token_stats_df: pd.DataFrame,
    group_by: str = 'dataset'
) -> pd.DataFrame:
    """
    Aggregate token statistics
    
    Args:
        token_stats_df: DataFrame with sample-level token statistics
        group_by: Aggregation dimension ('dataset', 'task_type', etc)
        
    Returns:
        Aggregated statistics DataFrame
    """
    agg_stats = token_stats_df.groupby(group_by).agg({
        'sample_id': 'count',
        'text_tokens': ['mean', 'std', 'min', 'max', 'sum'],
        'image_tokens': ['mean', 'std', 'min', 'max', 'sum'],
        'total_input_tokens': ['mean', 'std', 'min', 'max', 'sum'],
        'num_images': ['mean', 'sum']
    }).round(2)
    
    # Flatten column names
    agg_stats.columns = ['_'.join(col).strip('_') for col in agg_stats.columns]
    agg_stats = agg_stats.rename(columns={'sample_id_count': 'num_samples'})
    
    return agg_stats.reset_index()


def compute_token_based_costs(
    input_tokens_df: pd.DataFrame,
    output_tokens_df: pd.DataFrame,
    pricing_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Calculate actual costs based on token counts
    
    Args:
        input_tokens_df: Input token statistics
        output_tokens_df: Output token statistics  
        pricing_config: Pricing config {model_name: {'input': $/1M tokens, 'output': $/1M tokens}}
        
    Returns:
        Cost statistics DataFrame
    """
    # Merge input and output token statistics
    merged = output_tokens_df.merge(
        input_tokens_df[['sample_id', 'total_input_tokens']],
        on='sample_id',
        how='left'
    )
    
    # If no pricing config provided, use default values
    if pricing_config is None:
        pricing_config = {
            'InternVL2_5-78B': {'input': 5.0, 'output': 15.0},  # $/1M tokens
            'Qwen2.5-VL-72B-Instruct': {'input': 4.0, 'output': 12.0},
            'Qwen2.5-VL-32B-Instruct': {'input': 2.0, 'output': 6.0},
            'MiMo-VL-7B-RL': {'input': 0.5, 'output': 1.5},
            'Phi-3.5-Vision': {'input': 0.3, 'output': 1.0},
            'SmolVLM2': {'input': 0.2, 'output': 0.6},
            'llava_next_vicuna_7b': {'input': 0.3, 'output': 1.0},
        }
    
    # Calculate costs
    merged['input_cost'] = merged.apply(
        lambda row: (row['total_input_tokens'] / 1_000_000) * 
                    pricing_config.get(row['model'], {'input': 1.0})['input'],
        axis=1
    )
    
    merged['output_cost'] = merged.apply(
        lambda row: (row['output_tokens'] / 1_000_000) * 
                    pricing_config.get(row['model'], {'output': 3.0})['output'],
        axis=1
    )
    
    merged['total_cost'] = merged['input_cost'] + merged['output_cost']
    
    return merged
