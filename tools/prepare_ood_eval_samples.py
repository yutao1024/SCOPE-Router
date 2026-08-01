#!/usr/bin/env python3
"""
Prepare strictly verifiable OOD evaluation samples.

This script builds BENCHMARKS-compatible JSONL files for:

  - CLEVR val
  - GQA val balanced
  - VizWiz val

The default output is 500 samples per dataset, sampled deterministically with
light stratification over answer/question types. We deliberately use public
splits with local ground truth so that every sample can be verified before
model inference.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import shutil
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import tarfile
import zipfile


TASK_TYPE = "vqa_oe"

CLEVR_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
GQA_QUESTIONS_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"
GQA_IMAGES_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
VIZWIZ_VAL_IMAGES_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip"
VIZWIZ_ANNOTATIONS_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip"
COCO_TRAIN2014_URL = "http://images.cocodataset.org/zips/train2014.zip"
COCO_VAL2014_URL = "http://images.cocodataset.org/zips/val2014.zip"
VQACP2_TAR_URL = "https://data.lip6.fr/cadene/murel/vqacp2.tar.gz"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_file(
    url: str,
    output_path: Path,
    force: bool = False,
    no_check_certificate: bool = False,
    retries: int = 3,
    retry_wait_seconds: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f"  Reusing existing download: {output_path}")
        return output_path

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    print(f"  Downloading: {url}")
    print(f"       -> {output_path}")
    context = ssl._create_unverified_context() if no_check_certificate else None
    last_error = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            with urllib.request.urlopen(url, context=context) as response, tmp_path.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            last_error = None
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 503 or attempt >= retries:
                break
            print(f"  HTTP 503 from source; retrying in {retry_wait_seconds}s ({attempt}/{retries})")
            time.sleep(retry_wait_seconds)
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(f"  Download failed: {exc}; retrying in {retry_wait_seconds}s ({attempt}/{retries})")
            time.sleep(retry_wait_seconds)

    if last_error is not None:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Failed to download {url}: {last_error}. "
            "If this is VQA-CP v2 from data.lip6.fr, the upstream mirror is currently unavailable; "
            "try again later or pass a different --vqacp-tar-url / JSON mirror."
        ) from last_error
    tmp_path.replace(output_path)
    return output_path


def extract_zip(zip_path: Path, output_dir: Path, force: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinel = output_dir / f".extract_complete_{zip_path.name}"
    if sentinel.exists() and not force:
        print(f"  Reusing extracted archive: {output_dir}")
        return
    print(f"  Extracting: {zip_path} -> {output_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)
    sentinel.write_text(str(zip_path) + "\n", encoding="utf-8")


def extract_tar(tar_path: Path, output_dir: Path, force: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinel = output_dir / f".extract_complete_{tar_path.name}"
    if sentinel.exists() and not force:
        print(f"  Reusing extracted archive: {output_dir}")
        return
    print(f"  Extracting: {tar_path} -> {output_dir}")
    with tarfile.open(tar_path) as tf:
        tf.extractall(output_dir)
    sentinel.write_text(str(tar_path) + "\n", encoding="utf-8")


def download_and_extract_zip(
    url: str,
    archive_dir: Path,
    extract_dir: Path,
    force: bool = False,
    no_check_certificate: bool = False,
    retries: int = 3,
    retry_wait_seconds: int = 30,
) -> None:
    archive_name = url.rsplit("/", 1)[-1]
    archive_path = download_file(
        url,
        archive_dir / archive_name,
        force=force,
        no_check_certificate=no_check_certificate,
        retries=retries,
        retry_wait_seconds=retry_wait_seconds,
    )
    extract_zip(archive_path, extract_dir, force=force)


def download_and_extract_tar(
    url: str,
    archive_dir: Path,
    extract_dir: Path,
    force: bool = False,
    no_check_certificate: bool = False,
    retries: int = 3,
    retry_wait_seconds: int = 30,
) -> None:
    archive_name = url.rsplit("/", 1)[-1]
    archive_path = download_file(
        url,
        archive_dir / archive_name,
        force=force,
        no_check_certificate=no_check_certificate,
        retries=retries,
        retry_wait_seconds=retry_wait_seconds,
    )
    extract_tar(archive_path, extract_dir, force=force)


def first_existing_dir(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists() and path.is_dir():
            return path
    return None


def download_requested_datasets(args: argparse.Namespace) -> None:
    if not args.download:
        return

    selected = {item.strip().lower() for item in args.download_datasets.split(",") if item.strip()}
    if "all" in selected:
        selected = {"clevr", "gqa", "vizwiz", "vqacp"}
    download_root = Path(args.download_dir)
    archive_dir = download_root / "_archives"
    download_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Downloading requested OOD datasets")
    print("=" * 80)

    if "clevr" in selected:
        clevr_extract = download_root
        download_and_extract_zip(
            args.clevr_url,
            archive_dir,
            clevr_extract,
            force=args.download_force,
            no_check_certificate=args.download_no_check_certificate,
            retries=args.download_retries,
            retry_wait_seconds=args.download_retry_wait_seconds,
        )
        args.clevr_dir = args.clevr_dir or str(first_existing_dir([
            download_root / "CLEVR_v1.0",
            download_root / "CLEVR",
        ]) or (download_root / "CLEVR_v1.0"))

    if "gqa" in selected:
        gqa_root = download_root / "GQA"
        download_and_extract_zip(
            args.gqa_questions_url,
            archive_dir,
            gqa_root,
            force=args.download_force,
            no_check_certificate=args.download_no_check_certificate,
            retries=args.download_retries,
            retry_wait_seconds=args.download_retry_wait_seconds,
        )
        if args.download_gqa_images:
            download_and_extract_zip(
                args.gqa_images_url,
                archive_dir,
                gqa_root,
                force=args.download_force,
                no_check_certificate=args.download_no_check_certificate,
                retries=args.download_retries,
                retry_wait_seconds=args.download_retry_wait_seconds,
            )
        args.gqa_dir = args.gqa_dir or str(gqa_root)
        args.gqa_image_root = args.gqa_image_root or str(gqa_root / "images")

    if "vizwiz" in selected:
        vizwiz_root = download_root / "VizWiz"
        download_and_extract_zip(
            args.vizwiz_annotations_url,
            archive_dir,
            vizwiz_root,
            force=args.download_force,
            no_check_certificate=args.download_no_check_certificate,
            retries=args.download_retries,
            retry_wait_seconds=args.download_retry_wait_seconds,
        )
        download_and_extract_zip(
            args.vizwiz_val_images_url,
            archive_dir,
            vizwiz_root / "images",
            force=args.download_force,
            no_check_certificate=args.download_no_check_certificate,
            retries=args.download_retries,
            retry_wait_seconds=args.download_retry_wait_seconds,
        )
        args.vizwiz_dir = args.vizwiz_dir or str(vizwiz_root)
        args.vizwiz_image_root = args.vizwiz_image_root or str(vizwiz_root / "images")

    if "vqacp" in selected:
        vqacp_root = download_root / "VQA-CP-v2"
        vqacp_root.mkdir(parents=True, exist_ok=True)
        if args.vqacp_tar_url:
            download_and_extract_tar(
                args.vqacp_tar_url,
                archive_dir,
                vqacp_root,
                force=args.download_force,
                no_check_certificate=args.download_no_check_certificate,
                retries=args.download_retries,
                retry_wait_seconds=args.download_retry_wait_seconds,
            )
        elif not args.vqacp_questions_url or not args.vqacp_annotations_url:
            raise ValueError(
                "VQA-CP v2 download needs --vqacp-tar-url, or both "
                "--vqacp-questions-url and --vqacp-annotations-url."
            )
        else:
            download_file(
                args.vqacp_questions_url,
                vqacp_root / "vqacp_v2_test_questions.json",
                force=args.download_force,
                no_check_certificate=args.download_no_check_certificate,
                retries=args.download_retries,
                retry_wait_seconds=args.download_retry_wait_seconds,
            )
            download_file(
                args.vqacp_annotations_url,
                vqacp_root / "vqacp_v2_test_annotations.json",
                force=args.download_force,
                no_check_certificate=args.download_no_check_certificate,
                retries=args.download_retries,
                retry_wait_seconds=args.download_retry_wait_seconds,
            )
        args.vqacp_dir = args.vqacp_dir or str(vqacp_root)

    if args.download_coco:
        coco_root = download_root / "COCO"
        download_and_extract_zip(
            COCO_TRAIN2014_URL,
            archive_dir,
            coco_root,
            force=args.download_force,
            no_check_certificate=args.download_no_check_certificate,
            retries=args.download_retries,
            retry_wait_seconds=args.download_retry_wait_seconds,
        )
        download_and_extract_zip(
            COCO_VAL2014_URL,
            archive_dir,
            coco_root,
            force=args.download_force,
            no_check_certificate=args.download_no_check_certificate,
            retries=args.download_retries,
            retry_wait_seconds=args.download_retry_wait_seconds,
        )
        args.vqacp_image_root = args.vqacp_image_root or str(coco_root)


def require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def normalize_answer(answer: Any) -> str:
    return str(answer).strip()


def answer_type(answer: Any) -> str:
    value = normalize_answer(answer).lower()
    if value in {"yes", "no", "true", "false"}:
        return "yes_no"
    try:
        float(value)
        return "number"
    except ValueError:
        return "other"


def normalize_answer_type(value: Any) -> str:
    normalized = normalize_answer(value).lower().replace("-", "_").replace("/", "_")
    if normalized in {"yes_no", "yesno"}:
        return "yes_no"
    if normalized in {"unanswerable", "unsuitable"}:
        return "unanswerable"
    if normalized in {"number", "other"}:
        return normalized
    return normalized or "unknown"


def maybe_resolve(path: Path) -> str:
    try:
        return str(path.resolve())
    except FileNotFoundError:
        return str(path)


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def clean_dataset_name(name: str) -> str:
    return name.upper().replace("-", "_")


def stratified_sample(
    rows: List[Dict[str, Any]],
    n: int,
    rng: random.Random,
    key_fn: Callable[[Dict[str, Any]], str],
) -> List[Dict[str, Any]]:
    if n <= 0 or len(rows) <= n:
        return sorted(rows, key=lambda row: row["sample_id"])

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row) or "unknown")].append(row)

    for group_rows in groups.values():
        rng.shuffle(group_rows)

    total = len(rows)
    allocations: Dict[str, int] = {}
    fractions: List[Tuple[float, str]] = []
    for key, group_rows in groups.items():
        exact = n * len(group_rows) / total
        count = min(len(group_rows), int(exact))
        allocations[key] = count
        fractions.append((exact - count, key))

    selected_count = sum(allocations.values())
    for _, key in sorted(fractions, reverse=True):
        if selected_count >= n:
            break
        if allocations[key] < len(groups[key]):
            allocations[key] += 1
            selected_count += 1

    while selected_count < n:
        grew = False
        for key in sorted(groups):
            if selected_count >= n:
                break
            if allocations[key] < len(groups[key]):
                allocations[key] += 1
                selected_count += 1
                grew = True
        if not grew:
            break

    sampled: List[Dict[str, Any]] = []
    for key, count in allocations.items():
        sampled.extend(groups[key][:count])
    rng.shuffle(sampled)
    return sorted(sampled[:n], key=lambda row: row["sample_id"])


def add_selection_index(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for index, row in enumerate(rows):
        row["selection_index"] = index
    return rows


def verify_samples(rows: List[Dict[str, Any]], allow_missing_images: bool) -> Dict[str, Any]:
    errors = []
    missing_images = []
    duplicate_ids = [sid for sid, count in Counter(row.get("sample_id") for row in rows).items() if count > 1]
    if duplicate_ids:
        errors.append(f"Duplicate sample_id values: {duplicate_ids[:5]}")

    for row in rows:
        sid = row.get("sample_id", "<missing>")
        if not row.get("question") and not row.get("prompt"):
            errors.append(f"{sid}: missing question/prompt")
        if row.get("answer") in (None, ""):
            errors.append(f"{sid}: missing answer")
        if not row.get("eval_method"):
            errors.append(f"{sid}: missing eval_method")
        assets = row.get("assets") or []
        if not assets:
            errors.append(f"{sid}: missing image assets")
            continue
        for asset in assets:
            if not isinstance(asset, dict) or asset.get("type") != "image":
                errors.append(f"{sid}: unsupported asset {asset!r}; expected type=image")
                continue
            image_path = asset.get("path")
            if not image_path:
                errors.append(f"{sid}: image asset missing path")
                continue
            if not Path(image_path).exists():
                missing_images.append((sid, image_path))

    if missing_images and not allow_missing_images:
        preview = "; ".join(f"{sid}: {path}" for sid, path in missing_images[:5])
        errors.append(f"Missing image files ({len(missing_images)} total): {preview}")

    if errors:
        preview = "\n  - ".join(errors[:10])
        raise ValueError(f"Sample verification failed:\n  - {preview}")

    return {
        "num_samples": len(rows),
        "num_missing_images": len(missing_images),
        "answer_type_counts": dict(Counter(row.get("answer_type", "unknown") for row in rows)),
        "eval_method_counts": dict(Counter(row.get("eval_method", "unknown") for row in rows)),
    }


def clevr_question_file(root: Path, split: str) -> Path:
    candidates = [
        root / "questions" / f"CLEVR_{split}_questions.json",
        root / f"CLEVR_{split}_questions.json",
        root / f"{split}_questions.json",
    ]
    path = first_existing(candidates)
    if path is None:
        raise FileNotFoundError(
            "Could not find CLEVR questions file. Tried: "
            + ", ".join(str(item) for item in candidates)
        )
    return path


def load_clevr(root: Path, split: str) -> List[Dict[str, Any]]:
    question_path = clevr_question_file(root, split)
    data = read_json(question_path)
    questions = data.get("questions", data if isinstance(data, list) else None)
    if not isinstance(questions, list):
        raise ValueError(f"Unexpected CLEVR questions format: {question_path}")

    image_root = root / "images" / split
    rows = []
    for idx, item in enumerate(questions):
        if "answer" not in item or item.get("answer") is None:
            raise ValueError(
                f"CLEVR split {split!r} has no public answer at row {idx}. "
                "Use --clevr-split val for strict local verification."
            )
        image_filename = item.get("image_filename")
        image_path = image_root / image_filename if image_filename else Path("")
        source_id = str(item.get("question_index", idx))
        answer = normalize_answer(item["answer"])
        program = item.get("program") or []
        rows.append({
            "sample_id": f"CLEVR_{split.upper()}/{source_id}",
            "dataset": f"CLEVR_{split.upper()}",
            "task_type": TASK_TYPE,
            "modality": ["image", "text"],
            "prompt": item.get("question", "").strip(),
            "question": item.get("question", "").strip(),
            "assets": [{"type": "image", "path": maybe_resolve(image_path)}],
            "answer": answer,
            "answers": [answer],
            "answer_type": answer_type(answer),
            "question_type": program[-1].get("function", "unknown") if program else "unknown",
            "eval_method": "exact_match",
            "source_dataset": "CLEVR",
            "source_split": split,
            "source_id": source_id,
            "image_id": item.get("image_index"),
            "image_filename": image_filename,
        })
    return rows


def find_gqa_question_file(root: Path, split: str, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return require_file(explicit, "GQA questions file")
    candidates = [
        root / f"{split}_questions.json",
        root / "questions" / f"{split}_questions.json",
    ]
    if split == "val_balanced":
        candidates.extend([
            root / "val_balanced_questions.json",
            root / "questions" / "val_balanced_questions.json",
        ])
    path = first_existing(candidates)
    if path is None:
        raise FileNotFoundError(
            "Could not find GQA questions file. Tried: "
            + ", ".join(str(item) for item in candidates)
        )
    return path


def resolve_gqa_image(root: Path, image_root: Optional[Path], image_id: Any) -> Path:
    base_dirs = []
    if image_root is not None:
        base_dirs.append(image_root)
    base_dirs.extend([root / "images", root / "allImages", root])
    names = [f"{image_id}.jpg", f"{image_id}.jpeg", f"{image_id}.png"]
    candidates = [base / name for base in base_dirs for name in names]
    path = first_existing(candidates)
    return path if path is not None else candidates[0]


def gqa_type(item: Dict[str, Any]) -> str:
    types = item.get("types") or {}
    if isinstance(types, dict):
        return str(types.get("structural") or types.get("semantic") or "unknown")
    return "unknown"


def load_gqa(root: Path, split: str, questions_file: Optional[Path], image_root: Optional[Path]) -> List[Dict[str, Any]]:
    question_path = find_gqa_question_file(root, split, questions_file)
    data = read_json(question_path)
    if isinstance(data, dict) and "questions" in data:
        raw_questions = data["questions"]
        if isinstance(raw_questions, dict):
            items = list(raw_questions.items())
        else:
            items = [(str(item.get("question_id", idx)), item) for idx, item in enumerate(raw_questions)]
    elif isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, list):
        items = [(str(item.get("question_id", idx)), item) for idx, item in enumerate(data)]
    else:
        raise ValueError(f"Unexpected GQA questions format: {question_path}")

    dataset = clean_dataset_name(f"GQA_{split}")
    rows = []
    for idx, (question_id, item) in enumerate(items):
        if "answer" not in item or item.get("answer") is None:
            raise ValueError(
                f"GQA split {split!r} has no public answer at row {idx}. "
                "Use a public validation split such as val_balanced for strict local verification."
            )
        image_id = item.get("imageId", item.get("image_id"))
        answer = normalize_answer(item["answer"])
        question = item.get("question", "").strip()
        image_path = resolve_gqa_image(root, image_root, image_id)
        rows.append({
            "sample_id": f"{dataset}/{question_id}",
            "dataset": dataset,
            "task_type": TASK_TYPE,
            "modality": ["image", "text"],
            "prompt": question,
            "question": question,
            "assets": [{"type": "image", "path": maybe_resolve(image_path)}],
            "answer": answer,
            "answers": [answer],
            "answer_type": answer_type(answer),
            "question_type": gqa_type(item),
            "eval_method": "exact_match",
            "source_dataset": "GQA",
            "source_split": split,
            "source_id": str(question_id),
            "image_id": image_id,
        })
    return rows


def find_vizwiz_annotation_file(root: Path, split: str, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return require_file(explicit, "VizWiz annotation file")
    candidates = [
        root / f"{split}.json",
        root / "Annotations" / f"{split}.json",
        root / "annotations" / f"{split}.json",
        root / "vqa_data" / "Annotations" / f"{split}.json",
    ]
    path = first_existing(candidates)
    if path is not None:
        return path
    recursive_matches = [item for item in root.rglob(f"{split}.json")]
    if recursive_matches:
        return sorted(recursive_matches, key=lambda item: len(item.parts))[0]
    raise FileNotFoundError(
        "Could not find VizWiz annotation file. Tried: "
        + ", ".join(str(item) for item in candidates)
    )


def resolve_vizwiz_image(root: Path, split: str, image_root: Optional[Path], image_name: str) -> Path:
    base_dirs = []
    if image_root is not None:
        base_dirs.append(image_root)
    base_dirs.extend([
        root / "images" / split,
        root / "images",
        root / split,
        root,
    ])
    candidates = [base / image_name for base in base_dirs]
    path = first_existing(candidates)
    return path if path is not None else candidates[0]


def choose_vizwiz_answer(item: Dict[str, Any]) -> Tuple[str, List[str]]:
    raw_answers = item.get("answers") or []
    answers = []
    confident_answers = []
    for entry in raw_answers:
        if isinstance(entry, dict):
            answer = entry.get("answer")
            if answer is None:
                continue
            answer = normalize_answer(answer)
            answers.append(answer)
            if str(entry.get("answer_confidence", "")).lower() == "yes":
                confident_answers.append(answer)
        elif entry is not None:
            answers.append(normalize_answer(entry))

    pool = confident_answers or answers
    if pool:
        return Counter(pool).most_common(1)[0][0], answers or pool
    if item.get("answer") is not None:
        answer = normalize_answer(item["answer"])
        return answer, [answer]
    raise ValueError(f"VizWiz annotation has no answers: {item}")


def load_vizwiz(
    root: Path,
    split: str,
    annotations_file: Optional[Path],
    image_root: Optional[Path],
    include_unanswerable: bool = False,
) -> List[Dict[str, Any]]:
    ann_path = find_vizwiz_annotation_file(root, split, annotations_file)
    data = read_json(ann_path)
    items = data.get("annotations", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"Unexpected VizWiz annotations format: {ann_path}")

    dataset = f"VIZWIZ_{split.upper()}"
    rows = []
    for idx, item in enumerate(items):
        raw_answer_type = str(item.get("answer_type", "")).strip().lower()
        if raw_answer_type == "unanswerable" and not include_unanswerable:
            continue
        image_name = item.get("image")
        if not image_name:
            raise ValueError(f"VizWiz row {idx} has no image field")
        question = item.get("question", "").strip()
        answer, answers = choose_vizwiz_answer(item)
        image_path = resolve_vizwiz_image(root, split, image_root, image_name)
        source_id = Path(image_name).stem
        rows.append({
            "sample_id": f"{dataset}/{source_id}",
            "dataset": dataset,
            "task_type": TASK_TYPE,
            "modality": ["image", "text"],
            "prompt": question,
            "question": question,
            "assets": [{"type": "image", "path": maybe_resolve(image_path)}],
            "answer": answer,
            "answers": answers,
            "answer_type": normalize_answer_type(item.get("answer_type", answer_type(answer))),
            "question_type": normalize_answer_type(item.get("answer_type", "unknown")),
            "answerable": item.get("answerable"),
            "eval_method": "vqa_soft",
            "source_dataset": "VizWiz",
            "source_split": split,
            "source_id": source_id,
            "image_id": image_name,
        })
    return rows


def find_vqacp_file(root: Path, explicit: Optional[Path], names: Sequence[str], description: str) -> Path:
    if explicit is not None:
        return require_file(explicit, description)
    candidates = []
    for name in names:
        candidates.extend([root / name, root / "questions" / name, root / "annotations" / name])
    path = first_existing(candidates)
    if path is None:
        name_set = set(names)
        recursive_matches = [
            item for item in root.rglob("*.json")
            if item.name in name_set
        ]
        if recursive_matches:
            return sorted(recursive_matches, key=lambda item: len(item.parts))[0]
    if path is None:
        raise FileNotFoundError(
            f"Could not find {description}. Tried: " + ", ".join(str(item) for item in candidates)
        )
    return path


def as_vqa_list(data: Any, key: str) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and key in data:
        return data[key]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    raise ValueError(f"Unexpected VQA-style data format for key {key!r}")


def resolve_coco_image(image_root: Path, image_id: Any) -> Path:
    try:
        image_num = int(image_id)
        stem = f"{image_num:012d}"
    except (TypeError, ValueError):
        stem = str(image_id)

    candidates = []
    for split in ["train2014", "val2014", "test2015"]:
        names = [
            f"COCO_{split}_{stem}.jpg",
            f"{stem}.jpg",
            f"{image_id}.jpg",
        ]
        dirs = [
            image_root / split,
            image_root / "images" / split,
            image_root,
        ]
        candidates.extend(base / name for base in dirs for name in names)
    path = first_existing(candidates)
    return path if path is not None else candidates[0]


def choose_vqa_answer(annotation: Dict[str, Any]) -> Tuple[str, List[str]]:
    raw_answers = annotation.get("answers") or []
    answers = []
    for item in raw_answers:
        if isinstance(item, dict) and item.get("answer") is not None:
            answers.append(normalize_answer(item["answer"]))
        elif item is not None:
            answers.append(normalize_answer(item))
    answer = annotation.get("multiple_choice_answer")
    if answer is None and answers:
        answer = Counter(answers).most_common(1)[0][0]
    if answer is None:
        raise ValueError(f"VQA annotation has no answer: {annotation}")
    return normalize_answer(answer), answers or [normalize_answer(answer)]


def load_vqacp(
    root: Path,
    questions_file: Optional[Path],
    annotations_file: Optional[Path],
    image_root: Optional[Path],
) -> List[Dict[str, Any]]:
    q_path = find_vqacp_file(
        root,
        questions_file,
        [
            "vqacp_v2_test_questions.json",
            "vqa_cp_v2_test_questions.json",
            "test_questions.json",
        ],
        "VQA-CP v2 test questions file",
    )
    a_path = find_vqacp_file(
        root,
        annotations_file,
        [
            "vqacp_v2_test_annotations.json",
            "vqa_cp_v2_test_annotations.json",
            "test_annotations.json",
        ],
        "VQA-CP v2 test annotations file",
    )
    questions = as_vqa_list(read_json(q_path), "questions")
    annotations = as_vqa_list(read_json(a_path), "annotations")
    ann_by_qid = {str(item.get("question_id")): item for item in annotations}
    root_for_images = image_root or root

    rows = []
    for idx, item in enumerate(questions):
        qid = str(item.get("question_id", idx))
        if qid not in ann_by_qid:
            raise ValueError(f"VQA-CP question {qid} has no annotation in {a_path}")
        ann = ann_by_qid[qid]
        answer, answers = choose_vqa_answer(ann)
        image_id = item.get("image_id", ann.get("image_id"))
        image_path = resolve_coco_image(root_for_images, image_id)
        question = item.get("question", "").strip()
        rows.append({
            "sample_id": f"VQACP_V2_TEST/{qid}",
            "dataset": "VQACP_V2_TEST",
            "task_type": TASK_TYPE,
            "modality": ["image", "text"],
            "prompt": question,
            "question": question,
            "assets": [{"type": "image", "path": maybe_resolve(image_path)}],
            "answer": answer,
            "answers": answers,
            "answer_type": str(ann.get("answer_type", answer_type(answer))),
            "question_type": str(ann.get("question_type", "unknown")),
            "eval_method": "vqa_soft",
            "source_dataset": "VQA-CP v2",
            "source_split": "test",
            "source_id": qid,
            "image_id": image_id,
        })
    return rows


def parse_optional_path(value: Optional[str], base: Optional[Path] = None) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if base is not None and not path.is_absolute():
        path = base / path
    return path


def prepare_one(
    name: str,
    rows: List[Dict[str, Any]],
    samples_per_dataset: int,
    rng: random.Random,
    allow_missing_images: bool,
    output_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sampled = stratified_sample(
        rows,
        samples_per_dataset,
        rng,
        key_fn=lambda row: f"{row.get('answer_type', 'unknown')}::{row.get('question_type', 'unknown')}",
    )
    sampled = add_selection_index(sampled)
    verification = verify_samples(sampled, allow_missing_images=allow_missing_images)
    write_jsonl(output_path, sampled)
    summary = {
        "dataset": name,
        "available_samples": len(rows),
        "selected_samples": len(sampled),
        "output": str(output_path),
        "verification": verification,
        "source_split_counts": dict(Counter(row.get("source_split", "unknown") for row in sampled)),
    }
    return sampled, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OOD evaluation JSONL files with strict answer/image checks")
    parser.add_argument("--output-dir", default="OOD_BENCHMARKS", help="Output directory")
    parser.add_argument("--samples-per-dataset", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing-images", action="store_true",
                        help="Write samples even if image files are missing")

    parser.add_argument("--download", action="store_true",
                        help="Download requested raw datasets before sampling")
    parser.add_argument("--download-dir", default="RAW_OOD_DATA",
                        help="Directory for downloaded raw datasets")
    parser.add_argument("--download-datasets", default="clevr,gqa,vizwiz",
                        help="Comma list: clevr, gqa, vizwiz, vqacp, all. VQA-CP requires explicit URLs.")
    parser.add_argument("--download-force", action="store_true",
                        help="Re-download and re-extract archives even if files already exist")
    parser.add_argument("--download-no-check-certificate", action="store_true",
                        help="Disable TLS certificate verification for dataset downloads")
    parser.add_argument("--download-retries", type=int, default=3,
                        help="Download retry attempts for transient network/upstream errors")
    parser.add_argument("--download-retry-wait-seconds", type=int, default=30,
                        help="Seconds to wait between download retries")
    parser.add_argument("--download-gqa-images", action="store_true",
                        help="Download GQA images.zip. This is large; omit if images already exist elsewhere.")
    parser.add_argument("--download-coco", action="store_true",
                        help="Download COCO train2014 and val2014 images for VQA-CP v2")
    parser.add_argument("--clevr-url", default=CLEVR_URL)
    parser.add_argument("--gqa-questions-url", default=GQA_QUESTIONS_URL)
    parser.add_argument("--gqa-images-url", default=GQA_IMAGES_URL)
    parser.add_argument("--vizwiz-val-images-url", default=VIZWIZ_VAL_IMAGES_URL)
    parser.add_argument("--vizwiz-annotations-url", default=VIZWIZ_ANNOTATIONS_URL)
    parser.add_argument("--vqacp-tar-url", default=None,
                        help=f"VQA-CP v2 tar.gz URL, e.g. {VQACP2_TAR_URL}")
    parser.add_argument("--vqacp-questions-url", default=None,
                        help="Explicit URL for vqacp_v2_test_questions.json")
    parser.add_argument("--vqacp-annotations-url", default=None,
                        help="Explicit URL for vqacp_v2_test_annotations.json")

    parser.add_argument("--clevr-dir", default=None, help="CLEVR_v1.0 root directory")
    parser.add_argument("--clevr-split", default="val", help="CLEVR split with answers; use val for strict verification")

    parser.add_argument("--gqa-dir", default=None, help="GQA root directory")
    parser.add_argument("--gqa-split", default="val_balanced")
    parser.add_argument("--gqa-questions", default=None, help="Optional explicit GQA questions JSON")
    parser.add_argument("--gqa-image-root", default=None, help="Optional explicit GQA image root")

    parser.add_argument("--vizwiz-dir", default=None, help="VizWiz root directory")
    parser.add_argument("--vizwiz-split", default="val", help="VizWiz split with public answers")
    parser.add_argument("--vizwiz-annotations", default=None, help="Optional explicit VizWiz annotation JSON")
    parser.add_argument("--vizwiz-image-root", default=None, help="Optional explicit VizWiz image root")
    parser.add_argument("--vizwiz-include-unanswerable", action="store_true",
                        help="Include VizWiz unanswerable samples. Default excludes them for cleaner main experiments.")

    parser.add_argument("--vqacp-dir", default=None, help="VQA-CP v2 root directory")
    parser.add_argument("--vqacp-questions", default=None, help="Optional explicit VQA-CP v2 test questions JSON")
    parser.add_argument("--vqacp-annotations", default=None, help="Optional explicit VQA-CP v2 test annotations JSON")
    parser.add_argument("--vqacp-image-root", default=None, help="COCO train2014/val2014 image root")

    args = parser.parse_args()
    download_requested_datasets(args)

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)

    dataset_summaries = []
    all_rows: List[Dict[str, Any]] = []

    if args.clevr_dir:
        root = Path(args.clevr_dir)
        rows = load_clevr(root, split=args.clevr_split)
        sampled, summary = prepare_one(
            name=f"CLEVR_{args.clevr_split.upper()}",
            rows=rows,
            samples_per_dataset=args.samples_per_dataset,
            rng=rng,
            allow_missing_images=args.allow_missing_images,
            output_path=output_dir / TASK_TYPE / f"clevr_{args.clevr_split}_samples.jsonl",
        )
        all_rows.extend(sampled)
        dataset_summaries.append(summary)

    if args.gqa_dir:
        root = Path(args.gqa_dir)
        rows = load_gqa(
            root,
            split=args.gqa_split,
            questions_file=parse_optional_path(args.gqa_questions, base=root),
            image_root=parse_optional_path(args.gqa_image_root, base=root),
        )
        sampled, summary = prepare_one(
            name=clean_dataset_name(f"GQA_{args.gqa_split}"),
            rows=rows,
            samples_per_dataset=args.samples_per_dataset,
            rng=rng,
            allow_missing_images=args.allow_missing_images,
            output_path=output_dir / TASK_TYPE / f"gqa_{args.gqa_split}_samples.jsonl",
        )
        all_rows.extend(sampled)
        dataset_summaries.append(summary)

    if args.vizwiz_dir:
        root = Path(args.vizwiz_dir)
        rows = load_vizwiz(
            root,
            split=args.vizwiz_split,
            annotations_file=parse_optional_path(args.vizwiz_annotations, base=root),
            image_root=parse_optional_path(args.vizwiz_image_root, base=root),
            include_unanswerable=args.vizwiz_include_unanswerable,
        )
        sampled, summary = prepare_one(
            name=f"VIZWIZ_{args.vizwiz_split.upper()}",
            rows=rows,
            samples_per_dataset=args.samples_per_dataset,
            rng=rng,
            allow_missing_images=args.allow_missing_images,
            output_path=output_dir / TASK_TYPE / f"vizwiz_{args.vizwiz_split}_samples.jsonl",
        )
        all_rows.extend(sampled)
        dataset_summaries.append(summary)

    if args.vqacp_dir:
        root = Path(args.vqacp_dir)
        rows = load_vqacp(
            root,
            questions_file=parse_optional_path(args.vqacp_questions, base=root),
            annotations_file=parse_optional_path(args.vqacp_annotations, base=root),
            image_root=parse_optional_path(args.vqacp_image_root, base=root),
        )
        sampled, summary = prepare_one(
            name="VQACP_V2_TEST",
            rows=rows,
            samples_per_dataset=args.samples_per_dataset,
            rng=rng,
            allow_missing_images=args.allow_missing_images,
            output_path=output_dir / TASK_TYPE / "vqacp_v2_test_samples.jsonl",
        )
        all_rows.extend(sampled)
        dataset_summaries.append(summary)

    if not dataset_summaries:
        raise ValueError("No datasets requested. Pass at least one dataset dir or use --download.")

    all_rows = sorted(all_rows, key=lambda row: (row["dataset"], row["sample_id"]))
    all_path = output_dir / TASK_TYPE / "ood_all_samples.jsonl"
    write_jsonl(all_path, all_rows)

    manifest = {
        "name": "ood_eval_samples",
        "samples_per_dataset": args.samples_per_dataset,
        "seed": args.seed,
        "task_type": TASK_TYPE,
        "strict_verification": not args.allow_missing_images,
        "total_selected_samples": len(all_rows),
        "datasets": dataset_summaries,
        "combined_output": str(all_path),
        "vizwiz_include_unanswerable": bool(args.vizwiz_include_unanswerable),
        "notes": [
            "CLEVR uses val by default because CLEVR test has no public answers.",
            "GQA uses val_balanced by default because it has local answer labels.",
            "VizWiz uses val by default because train/val have public answers and test labels are hidden.",
            "VizWiz unanswerable samples are excluded by default for cleaner main experiments.",
            "VQA-CP v2 remains supported, but its historical lip6 mirror may be unavailable.",
        ],
    }
    manifest_path = output_dir / "ood_eval_samples_manifest.json"
    write_json(manifest_path, manifest)

    print("=" * 80)
    print("OOD evaluation samples prepared")
    print("=" * 80)
    for summary in dataset_summaries:
        print(
            f"  {summary['dataset']}: selected {summary['selected_samples']} "
            f"/ {summary['available_samples']} -> {summary['output']}"
        )
    print(f"  Combined: {all_path}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
