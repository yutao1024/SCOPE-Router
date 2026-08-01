#!/usr/bin/env python3
"""
Build VLM Router Benchmark from VLMEvalKit results.
Read results in a similar way to summarize_results.py, then convert them into the benchmark format.
"""

import argparse
import json
import os
import re
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml


def load_yaml(path):
    """Load a YAML config file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_models_from_config(config_path='config/models.yaml'):
    """Load model list from config file."""
    try:
        config = load_yaml(config_path)
        models = list(config['models'].keys())
        print(f"✓ Loaded {len(models)} models from config")
        return models
    except Exception as e:
        print(f"⚠️  Failed to load models.yaml; using default model list: {e}")
        # Default model list (backward compatibility)
        return [
            'InternVL2_5-78B', 'MiMo-VL-7B-RL', 'Phi-3.5-Vision',
            'Qwen2.5-VL-32B-Instruct', 'Qwen2.5-VL-72B-Instruct',
            'SmolVLM2', 'llava_next_vicuna_7b',
            'Gemma3-27B', 'Janus-Pro-1B', 'Janus-Pro-7B',
            'Kimi-VL-A3B-Thinking-2506', 'Pixtral-12B', 'Qianfan-VL-8B',
            'deepseek_vl2', 'deepseek_vl2_tiny',
            'GPT4o', 'GeminiFlash2-5'  # Added API models
        ]


def extract_model_dataset(filepath, valid_models=None):
    """Extract model name and dataset name from the file path.
    
    Args:
        filepath: Result file path
        valid_models: Valid model list (loaded from config)
    """
    basename = os.path.basename(filepath)
    filepath_str = str(filepath)
    
    # Remove common suffixes
    name = basename.replace('_openai_result.xlsx', '') \
                   .replace('_result.xlsx', '') \
                   .replace('_gpt-4o-mini.xlsx', '') \
                   .replace('_gpt-4o-mini_score.xlsx', '') \
                   .replace('_gpt-4o-mini_extract.xlsx', '') \
                   .replace('_auxmatch.xlsx', '') \
                   .replace('_results.xlsx', '') \
                   .replace('.xlsx', '') \
                   .replace('.pkl', '')
    
    # Extract model name from the path (first matching model directory)
    path_parts = Path(filepath).parts
    model = None
    
    # If valid_models is provided, use it; otherwise fall back to a hard-coded list (backward compatibility)
    if valid_models is None:
        valid_models = [
            'InternVL2_5-78B', 'MiMo-VL-7B-RL', 'Phi-3.5-Vision',
            'Qwen2.5-VL-32B-Instruct', 'Qwen2.5-VL-72B-Instruct',
            'SmolVLM2', 'llava_next_vicuna_7b',
            'Gemma3-27B', 'Janus-Pro-1B', 'Janus-Pro-7B',
            'Kimi-VL-A3B-Thinking-2506', 'Pixtral-12B', 'Qianfan-VL-8B',
            'deepseek_vl2', 'deepseek_vl2_tiny',
            'GPT4o', 'GeminiFlash2-5'  # Added API models
        ]
    
    for part in path_parts:
        # Check whether this path part is a target model name
        if part in valid_models:
            model = part
            break
    
    # If not found, fall back to directory name
    if model is None:
        # Look for a possible model directory
        parent_dir = os.path.basename(os.path.dirname(filepath))
        if parent_dir and parent_dir not in ['T20251025_G', 'T20251027_Gb55f788e']:
            model = parent_dir
        else:
            # Search upward
            grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
            if grandparent_dir and grandparent_dir not in ['T20251025_G', 'T20251027_Gb55f788e']:
                model = grandparent_dir
    
    # Extract dataset name from file name
    if name.startswith(model + '_'):
        dataset = name[len(model)+1:]
    else:
        dataset = name
    
    return model, dataset


def identify_dataset_type(filepath, dataset_name):
    """Identify dataset result file type."""
    basename = os.path.basename(filepath)
    
    # Math type
    if 'gpt-4o-mini.xlsx' in basename or 'gpt-4o-mini_score.xlsx' in basename:
        return 'Math'
    
    # MCQ type
    if '_openai_result.xlsx' in basename:
        return 'MCQ'
    
    # Y/N type
    if '_auxmatch.xlsx' in basename or '_auxmatch.pkl' in basename:
        return 'Y/N'
    
    # OCRBench type
    if 'OCRBench' in dataset_name and '_results' not in basename:
        return 'OCRBench'
    
    # VQA type
    return 'VQA'


def read_is_correct_from_file(result_file, dataset_name):
    """Read is_correct from a result file, and also try to extract prompt and image path."""
    try:
        if result_file.endswith('.xlsx'):
            df = pd.read_excel(result_file, engine='openpyxl')
        elif result_file.endswith('.pkl'):
            df = pd.read_pickle(result_file)
        else:
            return None
        
        # Identify sample_id column
        if 'sample_id' in df.columns:
            sample_id_col = 'sample_id'
        elif 'id' in df.columns:
            sample_id_col = 'id'
        elif 'index' in df.columns:
            sample_id_col = 'index'
        else:
            # Fall back to DataFrame index
            df = df.reset_index()
            sample_id_col = 'index'
        
        # Identify is_correct column
        if 'is_correct' in df.columns:
            is_correct_col = 'is_correct'
        elif 'score' in df.columns:
            is_correct_col = 'score'
        elif 'hit' in df.columns:
            is_correct_col = 'hit'
        else:
            return None
        
        # Extract prompt column (try common candidates)
        prompt_col = None
        for col in ['question', 'prompt', 'Question', 'Prompt', 'text', 'query']:
            if col in df.columns:
                prompt_col = col
                break
        
        # Extract image column (try common candidates)
        image_col = None
        for col in ['image', 'image_path', 'Image', 'image_paths', 'img_path', 'asset', 'assets']:
            if col in df.columns:
                image_col = col
                break
        
        # Build result DataFrame
        result_cols = [sample_id_col, is_correct_col]
        if prompt_col:
            result_cols.append(prompt_col)
        if image_col:
            result_cols.append(image_col)
        
        df_result = df[result_cols].copy()
        
        # Rename sample_id column
        df_result.rename(columns={sample_id_col: 'sample_id_raw'}, inplace=True)
        
        # Normalize sample_id
        df_result['sample_id'] = df_result['sample_id_raw'].apply(
            lambda x: normalize_sample_id(x, dataset_name)
        )
        
        # Normalize is_correct
        if is_correct_col != 'is_correct':
            df_result['is_correct'] = df_result[is_correct_col].astype(int)
        
        # Standardize prompt and image fields
        if prompt_col:
            df_result['prompt'] = df_result[prompt_col].fillna('').astype(str)
        else:
            df_result['prompt'] = ''
        
        if image_col:
            df_result['image_path'] = df_result[image_col].fillna('').astype(str)
        else:
            df_result['image_path'] = ''
        
        # Return required columns
        output_cols = ['sample_id', 'is_correct']
        if prompt_col:
            output_cols.append('prompt')
        if image_col:
            output_cols.append('image_path')
        
        return df_result[output_cols].copy()
            
    except Exception as e:
        print(f"  ⚠️  Failed to read file {result_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def normalize_sample_id(sample_id, dataset_name):
    """Normalize sample_id format."""
    if pd.isna(sample_id):
        return None
    
    sample_id = str(sample_id).strip()
    
    # If already in dataset/sample_id format, return as-is
    if '/' in sample_id:
        return sample_id
    
    # Otherwise, prepend dataset
    return f"{dataset_name}/{sample_id}"


def collect_results_from_vlmevalkit(work_dir, target_datasets, target_models, valid_models=None):
    """Collect results from the VLMEvalKit output directory, including prompt and image paths.
    
    Args:
        work_dir: VLMEvalKit output directory
        target_datasets: Target dataset list
        target_models: Target model list
        valid_models: Valid model list (used for path parsing)
    """
    print("="*80)
    print("📥 Collecting results from VLMEvalKit")
    print("="*80)
    
    work_dir = Path(work_dir)
    
    # Find all result files (recursively)
    result_files = []
    
    # Math type (prefer gpt-4o-mini judge/score files)
    math_files_judge = list(work_dir.glob("**/*_gpt-4o-mini.xlsx"))
    math_files_score = list(work_dir.glob("**/*_gpt-4o-mini_score.xlsx"))
    result_files.extend([('Math', str(f)) for f in math_files_judge])
    result_files.extend([('Math', str(f)) for f in math_files_score])
    
    # MCQ type
    mcq_files = list(work_dir.glob("**/*_openai_result.xlsx"))
    result_files.extend([('MCQ', str(f)) for f in mcq_files])
    
    # Y/N type
    yorn_files_xlsx = list(work_dir.glob("**/*_auxmatch.xlsx"))
    yorn_files_pkl = list(work_dir.glob("**/*_auxmatch.pkl"))
    result_files.extend([('Y/N', str(f)) for f in yorn_files_xlsx])
    result_files.extend([('Y/N', str(f)) for f in yorn_files_pkl])
    
    # VQA type (e.g., *_results.xlsx)
    vqa_files = list(work_dir.glob("**/*_results.xlsx"))
    result_files.extend([('VQA', str(f)) for f in vqa_files])
    
    # Other xlsx files (not included above)
    all_xlsx = list(work_dir.glob("**/*.xlsx"))
    seen_files = {str(f) for _, f in result_files}
    for f in all_xlsx:
        if str(f) not in seen_files:
            # Exclude scoring helper files
            if not any(x in str(f) for x in ['_acc.xlsx', '_score.xlsx', '_extract.xlsx']):
                result_files.append(('VQA', str(f)))
    
    print(f"\nFound {len(result_files)} result files")
    
    # Collect results (including prompt and image paths)
    all_results = defaultdict(lambda: defaultdict(dict))  # {sample_id: {model: is_correct}}
    sample_metadata = {}  # {sample_id: {'prompt': str, 'assets': [...]}}
    
    for dtype, fpath in tqdm(result_files, desc="Processing result files"):
        try:
            model, dataset = extract_model_dataset(fpath, valid_models=valid_models)
            
            # Filter to target models/datasets
            if model not in target_models:
                continue
            if dataset not in target_datasets:
                continue
            
            # Read is_correct + prompt + image path
            df_result = read_is_correct_from_file(fpath, dataset)
            if df_result is None:
                print(f"  ⚠️  Skipping {model} × {dataset}: cannot read is_correct")
                continue
            
            # Store results
            for _, row in df_result.iterrows():
                sid = row['sample_id']
                is_correct = int(row['is_correct'])
                all_results[sid][model] = is_correct
                
                # Store sample metadata (prompt, assets, tsv_index) only once per sample
                if sid not in sample_metadata:
                    prompt = row.get('prompt', '')
                    image_path = row.get('image_path', '')
                    tsv_index = row.get('tsv_index', None)
                    
                    assets = []
                    if image_path and str(image_path).strip() and str(image_path) != 'nan':
                        # Build assets list
                        if isinstance(image_path, str):
                            # Handle multiple paths separated by ';' or ','
                            paths = [p.strip() for p in image_path.replace(',', ';').split(';') if p.strip()]
                            for path in paths:
                                assets.append({
                                    'type': 'image',
                                    'uri': path
                                })
                        elif isinstance(image_path, (list, tuple)):
                            for path in image_path:
                                if path and str(path).strip() and str(path) != 'nan':
                                    assets.append({
                                        'type': 'image',
                                        'uri': str(path)
                                    })
                    
                    sample_metadata[sid] = {
                        'prompt': str(prompt) if prompt else '',
                        'assets': assets,
                        'tsv_index': tsv_index  # Save TSV index for later lookup
                    }
                
        except Exception as e:
            print(f"  ⚠️  Failed to process file {fpath}: {str(e)}")
            continue
    
    print(f"\n✓ Collection complete: {len(all_results)} samples")
    print(f"  ✓ Samples with prompt: {sum(1 for m in sample_metadata.values() if m.get('prompt', '').strip())}")
    print(f"  ✓ Samples with assets: {sum(1 for m in sample_metadata.values() if m.get('assets', []))}")
    
    return all_results, sample_metadata


def find_tsv_file(dataset_name, tsv_base_dir):
    """Find a TSV file for a dataset, trying multiple naming conventions."""
    tsv_base_path = Path(tsv_base_dir)
    
    # Try multiple possible TSV filename formats
    possible_names = [
        f"{dataset_name}.tsv",           # original
        f"{dataset_name.upper()}.tsv",   # uppercase
        f"{dataset_name.replace('_', '')}.tsv",  # remove underscores
        # special mapping
        dataset_name.replace('_', '').upper() + '.tsv',
    ]
    
    # Try direct matches first
    for name in possible_names:
        tsv_path = tsv_base_path / name
        if tsv_path.exists():
            return tsv_path
    
    # If none found, try fuzzy match
    if tsv_base_path.exists():
        dataset_lower = dataset_name.lower().replace('_', '')
        for tsv_file in tsv_base_path.glob("*.tsv"):
            if tsv_file.stem.lower().replace('_', '') == dataset_lower:
                return tsv_file
    
    return None


def build_benchmark_structure(all_results, sample_metadata, output_dir, target_datasets, target_models, tsv_base_dir="/root/LMUData"):
    """Build benchmark directory structure with full prompt/assets info (including TSV image references)."""
    print("\n" + "="*80)
    print("🏗️  Building benchmark structure")
    print("="*80)
    
    output_dir = Path(output_dir)
    tsv_base_path = Path(tsv_base_dir)
    
    # Create directories
    benchmarks_dir = output_dir / "BENCHMARKS"
    oracle_dir = output_dir / "ORACLE" / "score"
    splits_dir = output_dir / "SPLITS"
    
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset -> task type mapping (simplified)
    dataset_to_task = {
        'MMMU_DEV_VAL': 'vqa_mc',
        'MathVista_MINI': 'math',
        'MathVision_MINI': 'math',
        'MathVerse_MINI': 'math',
        'MMBench_DEV_EN_V11': 'vqa_mc',
        'RealWorldQA': 'vqa_oe',
        'MMStar': 'vqa_mc',
        'HallusionBench': 'vqa_mc',
        'TextVQA_VAL': 'vqa_oe',
        'ChartQA_TEST': 'ocr_qa',
        'DocVQA_VAL': 'ocr_qa',
        'InfoVQA_VAL': 'ocr_qa',
        'AI2D_TEST': 'vqa_mc',
        'OCRBench': 'ocr_qa'
    }
    
    # Group by dataset
    dataset_samples = defaultdict(lambda: defaultdict(list))
    
    # Preload TSV mappings for all datasets (avoid repeated reads; resolve index vs row-number differences)
    print("  Preloading TSV mappings...")
    tsv_mappings = {}
    
    for dataset in target_datasets:
        tsv_path = find_tsv_file(dataset, tsv_base_dir)
        if tsv_path and tsv_path.exists():
            try:
                # Only the lightweight ID columns are needed to map sample IDs to
                # TSV row numbers. Avoid reading the base64 image column here; on
                # large TSVs it can consume enough memory for the process to be
                # killed before benchmark files are written.
                df_tsv = pd.read_csv(
                    tsv_path,
                    sep='\t',
                    dtype=str,
                    usecols=lambda col: col in {'index', 'id'}
                )
                
                # Build a mapping: sample_id -> row number
                mapping = {}
                
                # Choose mapping column based on dataset specifics
                if dataset == 'HallusionBench' and 'index' in df_tsv.columns:
                    # HallusionBench: index column contains the tail part of sample_id
                    for row_idx, row in df_tsv.iterrows():
                        mapping[row['index']] = row_idx
                elif dataset == 'MMMU_DEV_VAL' and 'id' in df_tsv.columns:
                    # MMMU_DEV_VAL: id column contains the tail part of sample_id
                    for row_idx, row in df_tsv.iterrows():
                        mapping[row['id']] = row_idx
                elif 'index' in df_tsv.columns:
                    # Other datasets: general handling, index value -> row number
                    for row_idx, row in df_tsv.iterrows():
                        # index column may contain integer IDs
                        index_val = row['index']
                        mapping[str(index_val)] = row_idx  # string key
                        try:
                            mapping[int(index_val)] = row_idx  # int key
                        except (ValueError, TypeError):
                            pass
                
                if mapping:
                    tsv_mappings[dataset] = {
                        'path': tsv_path,
                        'mapping': mapping,
                        'total_rows': len(df_tsv),
                    }
                    print(f"    ✓ {dataset}: {len(mapping)} mappings (TSV rows: {len(df_tsv)})")
            except Exception as e:
                print(f"    ⚠️  Failed to load {dataset}: {e}")
    
    for sid, model_results in all_results.items():
        # Parse sample_id: dataset/id
        if '/' in sid:
            dataset, sample_id = sid.split('/', 1)
        else:
            # For sample_ids without '/', try extracting dataset from prefix
            # e.g., mmmu_dev_000000 -> MMMU_DEV
            for target_ds in target_datasets:
                if sid.lower().startswith(target_ds.lower().replace('_', '')):
                    dataset = target_ds
                    sample_id = sid
                    break
            else:
                continue
        
        if dataset not in target_datasets:
            continue
        
        # Sample metadata
        metadata = sample_metadata.get(sid, {})
        prompt = metadata.get('prompt', '')
        assets = metadata.get('assets', [])
        tsv_index = metadata.get('tsv_index', None)
        
        # Prefer TSV image references whenever possible. Raw VLMEvalKit result
        # files may contain relative image paths such as "1_1.jpg" that are not
        # resolvable from this benchmark root, while TSV files contain the
        # actual base64 image payloads used by the feature extractor.
        assets_valid = []
        tsv_path = find_tsv_file(dataset, tsv_base_dir)
        if tsv_path:
            # Prefer tsv_index extracted from result files.
            if tsv_index is not None:
                try:
                    index_value = int(tsv_index)
                    assets_valid = [{
                        'type': 'image_tsv',
                        'tsv_file': str(tsv_path),
                        'index': index_value,
                        'description': f'Base64 encoded image from {tsv_path.name} at index {index_value}'
                    }]
                except (ValueError, TypeError):
                    pass

            # Otherwise, look up the normalized sample_id tail in the TSV mapping.
            if not assets_valid and dataset in tsv_mappings:
                tsv_info = tsv_mappings[dataset]
                tsv_path = tsv_info['path']
                mapping = tsv_info['mapping']

                row_num = None
                if sample_id in mapping:
                    row_num = mapping[sample_id]
                else:
                    try:
                        sample_id_int = int(sample_id)
                        if sample_id_int in mapping:
                            row_num = mapping[sample_id_int]
                    except (ValueError, TypeError):
                        pass

                if row_num is not None:
                    assets_valid = [{
                        'type': 'image_tsv',
                        'tsv_file': str(tsv_path),
                        'index': int(row_num),
                        'description': f'Base64 encoded image from {tsv_path.name} at row {row_num}'
                    }]

        # If no TSV reference can be built, keep image assets as-is.
        if not assets_valid:
            for asset in assets:
                if asset.get('type') == 'image':
                    uri = asset.get('uri', '')
                    if uri and str(uri).strip() and str(uri) != 'nan':
                        assets_valid.append(asset)
        
        # Use validated assets
        assets = assets_valid
        
        # Build sample record (with full prompt and assets)
        sample_record = {
            'sample_id': sid,
            'dataset': dataset,
            'task_type': dataset_to_task.get(dataset, 'unknown'),
            'modality': ['image', 'text'],
            'prompt': prompt,  # Full prompt text
            'assets': assets   # Image assets list (may include TSV references)
        }
        
        dataset_samples[dataset][sid] = sample_record
    
    # Save BENCHMARKS files
    print("\n📁 Saving BENCHMARKS files...")
    for dataset in target_datasets:
        if dataset not in dataset_samples:
            print(f"  ⚠️  {dataset}: no samples")
            continue
        
        # Determine task directory
        task_type = dataset_to_task.get(dataset, 'unknown')
        task_dir = benchmarks_dir / task_type
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Save samples.jsonl
        samples_file = task_dir / f"{dataset.lower()}_samples.jsonl"
        with open(samples_file, 'w', encoding='utf-8') as f:
            for sid, sample in dataset_samples[dataset].items():
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"  ✓ {dataset}: {len(dataset_samples[dataset])} samples")
    
    # Build scoring data (ORACLE/score)
    print("\n📊 Building scoring data...")
    score_records = []
    
    for sid, model_results in all_results.items():
        if '/' in sid:
            dataset, _ = sid.split('/', 1)
        else:
            continue
        
        if dataset not in target_datasets:
            continue
        
        for model, is_correct in model_results.items():
            score_records.append({
                'sample_id': sid,
                'model_id': model,
                'dataset': dataset,
                'quality_raw': int(is_correct),
                'quality': float(is_correct),
                'cost': 0.0  # To be filled from pricing.yaml later
            })
    
    # Save Parquet by dataset
    score_df = pd.DataFrame(score_records)
    for dataset in target_datasets:
        dataset_scores = score_df[score_df['dataset'] == dataset]
        if len(dataset_scores) > 0:
            output_file = oracle_dir / f"{dataset.lower()}.parquet"
            dataset_scores[['sample_id', 'model_id', 'quality_raw', 'quality']].to_parquet(
                output_file, index=False
            )
            print(f"  ✓ {dataset}: {len(dataset_scores)} score records")
    
    # Generate data splits (70/10/20: train/dev/test)
    print("\n📋 Generating data splits...")
    all_sample_ids = list(all_results.keys())
    np.random.seed(42)
    np.random.shuffle(all_sample_ids)
    
    n_total = len(all_sample_ids)
    n_train = int(n_total * 0.7)
    n_dev = int(n_total * 0.1)
    
    train_ids = all_sample_ids[:n_train]
    dev_ids = all_sample_ids[n_train:n_train+n_dev]
    test_ids = all_sample_ids[n_train+n_dev:]
    
    # Delete old test_id and test_ood files (if any)
    old_splits = ['test_id', 'test_ood']
    for old_split in old_splits:
        old_file = splits_dir / f"{old_split}.jsonl"
        if old_file.exists():
            old_file.unlink()
            print(f"  🗑️  Deleted old file: {old_split}.jsonl")
    
    for split_name, split_ids in [
        ('train', train_ids),
        ('dev', dev_ids),
        ('test', test_ids)
    ]:
        split_file = splits_dir / f"{split_name}.jsonl"
        with open(split_file, 'w', encoding='utf-8') as f:
            for sid in split_ids:
                f.write(json.dumps({'sample_id': sid}, ensure_ascii=False) + '\n')
        print(f"  ✓ {split_name}: {len(split_ids)} samples")
    
    return dataset_samples, score_df


def main():
    parser = argparse.ArgumentParser(description="Build benchmark from VLMEvalKit results")
    parser.add_argument('--vlmevalkit_dir', 
                       default='/opt/data/private/hzhcode/VLMEvalKit/lbj_outputs',
                       help='VLMEvalKit results directory')
    parser.add_argument('--output_dir', default='.', help='Benchmark output directory')
    parser.add_argument('--config_dir', default='config', help='Config directory')
    parser.add_argument('--tsv_base_dir', default='/root/LMUData', 
                       help='Base directory for TSV files (default: /root/LMUData)')
    
    args = parser.parse_args()
    
    # Target datasets
    target_datasets = [
        'MMMU_DEV_VAL', 'MathVista_MINI', 'MathVision_MINI', 'MathVerse_MINI',
        'MMBench_DEV_EN_V11', 'RealWorldQA', 'MMStar', 'HallusionBench',
        'TextVQA_VAL', 'ChartQA_TEST', 'DocVQA_VAL', 'InfoVQA_VAL',
        'AI2D_TEST', 'OCRBench'
    ]
    
    # Load model list from config (supports dynamic extension)
    models_config_path = Path(args.config_dir) / 'models.yaml'
    print(f"\nLoading model config: {models_config_path}")
    all_models = load_models_from_config(str(models_config_path))
    
    # Target models (loaded from config, with optional exclusions)
    # Note: models without outputs in the VLMEvalKit directory will be filtered out automatically.
    exclude_models = ['Ovis2.5-9B']  # Configurable exclusion list
    target_models = [m for m in all_models if m not in exclude_models]
    
    print(f"Target model count: {len(target_models)}")
    print(f"Model list: {', '.join(target_models)}")
    print()
    
    # 1. Collect results (including prompt and assets)
    all_results, sample_metadata = collect_results_from_vlmevalkit(
        args.vlmevalkit_dir, target_datasets, target_models, valid_models=all_models
    )
    
    # 2. Build benchmark structure (including TSV image references)
    dataset_samples, score_df = build_benchmark_structure(
        all_results, sample_metadata, args.output_dir, target_datasets, target_models,
        tsv_base_dir=args.tsv_base_dir
    )
    
    print("\n" + "="*80)
    print("✅ Benchmark build complete!")
    print("="*80)


if __name__ == '__main__':
    main()
