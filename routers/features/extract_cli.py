#!/usr/bin/env python3
"""
Feature extraction CLI - extract features independently from router training

Usage:
    python routers/features/extract_cli.py \
        --dataset_dir . \
        --text_encoder BAAI/bge-m3 \
        --vision_encoder openai/clip-vit-base-patch32 \
        --fusion_method concat \
        --output_dir EMBEDDINGS/fused

Supports ablation experiments:
    # Different text encoders
    python routers/features/extract_cli.py --text_encoder BAAI/bge-large-en-v1.5
    python routers/features/extract_cli.py --text_encoder intfloat/e5-large-v2
    
    # Different vision encoders
    python routers/features/extract_cli.py --vision_encoder google/siglip-base-patch16-224
    python routers/features/extract_cli.py --vision_encoder facebook/dinov2-base
    
    # Different fusion methods
    python routers/features/extract_cli.py --fusion_method concat_with_interaction
"""

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routers.features import FeatureExtractor, extract_all_features
from routers.features.encoders_registry import (
    AVAILABLE_TEXT_ENCODERS,
    AVAILABLE_VISION_ENCODERS,
    list_available_encoders
)


def resolve_sample_asset_paths(samples: list, dataset_dir: Path) -> list:
    """Resolve relative asset paths against dataset_dir for external dataset roots."""
    for sample in samples:
        assets = sample.get("assets", [])
        if not isinstance(assets, list):
            continue
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            for key in ["tsv_file", "uri", "path"]:
                value = asset.get(key)
                if not value:
                    continue
                path = Path(str(value))
                if not path.is_absolute():
                    asset[key] = str((dataset_dir / path).resolve())
    return samples


def load_samples(benchmarks_dir: Path, dataset_dir: Path = None) -> list:
    """Load all samples."""
    samples = []
    
    for task_dir in benchmarks_dir.iterdir():
        if not task_dir.is_dir():
            continue
        
        for samples_file in task_dir.glob("*_samples.jsonl"):
            with open(samples_file, 'r', encoding='utf-8') as f:
                for line in f:
                    samples.append(json.loads(line))
    if dataset_dir is not None:
        samples = resolve_sample_asset_paths(samples, dataset_dir)
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Extract text and vision features (decoupled from router training)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default encoders
  python routers/features/extract_cli.py --dataset_dir .

  # Ablation: different text encoders
  python routers/features/extract_cli.py --text_encoder BAAI/bge-large-en-v1.5
  python routers/features/extract_cli.py --text_encoder intfloat/e5-large-v2

  # Ablation: different vision encoders
  python routers/features/extract_cli.py --vision_encoder google/siglip-base-patch16-224
  python routers/features/extract_cli.py --vision_encoder facebook/dinov2-base

  # List all available encoders
  python routers/features/extract_cli.py --list_encoders
        """
    )
    
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=".",
        help="Dataset root directory"
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="BAAI/bge-m3",
        help="Text encoder (default: BAAI/bge-m3)"
    )
    parser.add_argument(
        "--vision_encoder",
        type=str,
        default="facebook/dinov2-base",
        help="Vision encoder (default: facebook/dinov2-base)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: dataset_dir/EMBEDDINGS)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device (default: auto)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--skip_text",
        action="store_true",
        help="Skip text feature extraction (vision only)"
    )
    parser.add_argument(
        "--skip_vision",
        action="store_true",
        help="Skip vision feature extraction (text only)"
    )
    parser.add_argument(
        "--list_encoders",
        action="store_true",
        help="List all available encoders"
    )
    
    args = parser.parse_args()
    
    # List available encoders
    if args.list_encoders:
        list_available_encoders()
        return
    
    dataset_dir = Path(args.dataset_dir)
    benchmarks_dir = dataset_dir / "BENCHMARKS"
    
    if not benchmarks_dir.exists():
        print(f"❌ BENCHMARKS directory not found: {benchmarks_dir}")
        print("   Please run build_benchmark_from_vlmevalkit.py to build the benchmark first.")
        return
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = dataset_dir / "EMBEDDINGS"
    else:
        output_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("🔍 Feature extraction tool (decoupled from router training)")
    print("=" * 80)
    print(f"Dataset dir: {dataset_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Text encoder: {args.text_encoder}")
    print(f"Vision encoder: {args.vision_encoder}")
    print("Note: feature fusion happens during router training")
    
    # Load samples
    print("\n📚 Loading samples...")
    samples = load_samples(benchmarks_dir, dataset_dir)
    print(f"  ✓ Total: {len(samples)} samples")
    
    if len(samples) == 0:
        print("❌ No samples found!")
        return
    
    # Text-only extraction
    if args.skip_vision:
        print("\n📝 Extracting text features only...")
        from routers.features import TextEncoder
        
        text_encoder = TextEncoder(
            model_name=args.text_encoder,
            device=args.device,
            batch_size=args.batch_size
        )
        
        texts = [s.get('prompt', '') for s in samples]
        text_embeddings = text_encoder.extract(texts)
        
        text_df = pd.DataFrame({
            'sample_id': [s['sample_id'] for s in samples],
            'embedding': list(text_embeddings)
        })
        
        # Save
        text_output_dir = dataset_dir / "EMBEDDINGS" / "text"
        text_output_dir.mkdir(parents=True, exist_ok=True)
        
        text_name = args.text_encoder.split('/')[-1]
        output_file = text_output_dir / f"{text_name}.parquet"
        text_df.to_parquet(output_file, index=False)
        
        print(f"  ✓ Text features saved: {output_file}")
        print(f"  Dim: {text_encoder.dimension}, samples: {len(text_df)}")
        
        return
    
    # Vision-only extraction
    if args.skip_text:
        print("\n🖼️  Extracting vision features only...")
        from routers.features import VisionEncoder
        
        vision_encoder = VisionEncoder(
            model_name=args.vision_encoder,
            device=args.device
        )
        
        vision_embeddings = vision_encoder.extract_from_samples(samples, batch_size=args.batch_size)
        
        vision_df = pd.DataFrame({
            'sample_id': [s['sample_id'] for s in samples],
            'embedding': list(vision_embeddings)
        })
        
        # Save
        vision_output_dir = dataset_dir / "EMBEDDINGS" / "vision"
        vision_output_dir.mkdir(parents=True, exist_ok=True)
        
        vision_name = args.vision_encoder.split('/')[-1]
        output_file = vision_output_dir / f"{vision_name}.parquet"
        vision_df.to_parquet(output_file, index=False)
        
        print(f"  ✓ Vision features saved: {output_file}")
        print(f"  Dim: {vision_encoder.dimension}, samples: {len(vision_df)}")
        
        return
    
    # Extract text and vision features (saved separately)
    print("\n📊 Extracting features (text and vision saved separately)...")
    
    # Create extractor
    extractor = FeatureExtractor(
        text_encoder=args.text_encoder,
        vision_encoder=args.vision_encoder,
        device=args.device,
        batch_size=args.batch_size
    )
    
    print(f"\n{extractor}")
    
    # Extract features
    result = extractor.extract_from_samples(samples, vision_batch_size=args.batch_size)
    
    # Save (text and vision separately)
    extractor.save_features(
        result['text_df'],
        result['vision_df'],
        output_dir
    )
    
    print("\n" + "=" * 80)
    print("✅ Feature extraction complete!")
    print("=" * 80)
    print("\nFeature file layout:")
    print(f"  {output_dir}/text/{args.text_encoder.split('/')[-1]}.parquet")
    print(f"  {output_dir}/vision/{args.vision_encoder.split('/')[-1]}.parquet")
    print("\n💡 Notes:")
    print("  - Text and vision features are saved separately")
    print("  - Feature fusion happens during router training (you can choose fusion methods flexibly)")
    print("\nAblation suggestions:")
    print(f"  - Text encoders: {', '.join(list(AVAILABLE_TEXT_ENCODERS.keys()))}")
    print(f"  - Vision encoders: {', '.join(list(AVAILABLE_VISION_ENCODERS.keys()))}")


if __name__ == "__main__":
    import pandas as pd  # imported only when needed
    main()
