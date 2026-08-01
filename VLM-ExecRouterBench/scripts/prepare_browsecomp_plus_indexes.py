import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "Tevatron/browsecomp-plus-indexes"
DEFAULT_OUT = ROOT / "indexes"
INDEX_ALIASES = {
    "bm25": "bm25",
    "qwen3-embedding-0.6B": "qwen3-embedding-0.6b",
    "qwen3-embedding-0.6b": "qwen3-embedding-0.6b",
    "qwen3-embedding-4B": "qwen3-embedding-4b",
    "qwen3-embedding-4b": "qwen3-embedding-4b",
    "qwen3-embedding-8B": "qwen3-embedding-8b",
    "qwen3-embedding-8b": "qwen3-embedding-8b",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official BrowseComp-Plus retriever indexes.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        choices=sorted(INDEX_ALIASES),
        help="Index folder to download. Defaults to bm25. Can be passed multiple times.",
    )
    args = parser.parse_args()
    requested = args.include or ["bm25"]
    includes = [INDEX_ALIASES[item] for item in requested]

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = []
    for name in dict.fromkeys(includes + requested):
        patterns.append(f"{name}/*")

    local_dir = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(out_dir),
            allow_patterns=patterns,
        )
    ).resolve()
    try:
        display_path = local_dir.relative_to(ROOT)
    except ValueError:
        display_path = local_dir
    print(f"[saved] {display_path} includes={','.join(dict.fromkeys(includes))}")


if __name__ == "__main__":
    main()
