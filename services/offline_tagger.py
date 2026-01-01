from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.io import readable_path, log_join
from utils.dataset import split_tags, join_tags


DEFAULT_MODEL_ID = "SmilingWolf/wd-swinv2-tagger-v3"
DEFAULT_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

DEFAULT_GENERAL_THRESHOLD = 0.35
DEFAULT_CHARACTER_THRESHOLD = 0.85

CATEGORY_CHARACTER = 3
CATEGORY_RATING = 9

_MODEL_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_DOWNLOAD_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.csv"]


@dataclass
class TaggerOptions:
    dataset_path: Path
    recursive: bool
    image_exts: List[str]
    model_id: str
    device: str
    batch_size: int
    general_threshold: float
    character_threshold: float
    include_character: bool
    include_rating: bool
    replace_underscore: bool
    write_mode: str
    preview_only: bool
    preview_limit: int
    limit: int
    max_tags: int
    skip_empty: bool
    local_only: bool


def _parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(val: Any, default: int) -> int:
    try:
        return int(val)
    except Exception:
        return default


def _parse_float(val: Any, default: float) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _parse_exts(raw: Any) -> List[str]:
    if isinstance(raw, list):
        values = raw
    else:
        values = str(raw or "").split(",")
    out = []
    for token in values:
        val = str(token).strip().lower()
        if not val:
            continue
        if not val.startswith("."):
            val = "." + val
        out.append(val)
    return out or list(DEFAULT_IMAGE_EXTS)


def _iter_images(root: Path, recursive: bool, exts: List[str]) -> List[Path]:
    exts_set = {e.lower() for e in exts}
    iterator = root.rglob("*") if recursive else root.iterdir()
    out = []
    for p in iterator:
        if not p.is_file():
            continue
        if p.suffix.lower() in exts_set:
            out.append(p)
    out.sort(key=lambda p: p.as_posix().lower())
    return out


def _chunked(seq: List[Path], size: int):
    size = max(1, size)
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _load_tag_csv(path: Path) -> List[Tuple[str, Optional[int]]]:
    import csv

    rows: List[List[str]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = [row for row in reader if row]
    except Exception:
        return []
    if not rows:
        return []
    header = [h.strip().lower() for h in rows[0]]
    has_header = "name" in header
    idx_name = header.index("name") if "name" in header else 1
    idx_cat = header.index("category") if "category" in header else None
    data_rows = rows[1:] if has_header else rows
    out = []
    for row in data_rows:
        if len(row) <= idx_name:
            continue
        name = row[idx_name].strip()
        if not name:
            continue
        cat = None
        if idx_cat is not None and len(row) > idx_cat:
            try:
                cat = int(row[idx_cat].strip())
            except Exception:
                cat = None
        out.append((name, cat))
    return out


def _load_tag_metadata(model_path: Path) -> List[Tuple[str, Optional[int]]]:
    candidates = ["selected_tags.csv", "tags.csv"]
    if not model_path.exists():
        return []
    for name in candidates:
        path = model_path / name
        if path.exists():
            return _load_tag_csv(path)
    return []


def _has_safetensors(path: Path) -> bool:
    return any(path.rglob("*.safetensors"))


def _ensure_model_local(model_id: str, local_only: bool) -> Path:
    model_path = Path(model_id)
    if model_path.exists():
        if not _has_safetensors(model_path):
            raise RuntimeError("Safetensors weights not found in the local model folder.")
        return model_path

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("Missing huggingface_hub; cannot download model files.") from exc

    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            allow_patterns=_DOWNLOAD_PATTERNS,
            local_files_only=local_only,
        )
    except Exception as exc:
        if local_only:
            raise RuntimeError("Model files not found locally. Disable local-only to download.") from exc
        raise RuntimeError("Failed to download model files from Hugging Face.") from exc

    model_path = Path(local_dir)
    if not _has_safetensors(model_path):
        raise RuntimeError("Safetensors weights not found in downloaded files.")
    return model_path


def _resolve_device(requested: str, torch_module) -> Tuple[str, Optional[str]]:
    requested = (requested or "auto").strip().lower()
    if requested in {"auto", ""}:
        if torch_module.cuda.is_available():
            return "cuda", None
        return "cpu", None
    if requested == "cuda":
        if torch_module.cuda.is_available():
            return "cuda", None
        return "cpu", "CUDA not available; falling back to CPU."
    return requested, None


def _load_model_bundle(model_id: str, device: str, local_only: bool):
    cache_key = (model_id, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except Exception as exc:
        raise RuntimeError(
            "Missing deps for offline tagger. Install torch and transformers first."
        ) from exc

    resolved_device, warn = _resolve_device(device, torch)
    model_path = _ensure_model_local(model_id, local_only)
    processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_safetensors=True,
    )
    model.eval()
    model.to(resolved_device)

    num_labels = int(getattr(model.config, "num_labels", 0)) or len(getattr(model.config, "id2label", {}))
    id2label = getattr(model.config, "id2label", {}) or {}
    labels = []
    for i in range(num_labels):
        label = id2label.get(i) or id2label.get(str(i)) or f"tag_{i}"
        labels.append(str(label))

    categories: Optional[List[Optional[int]]] = None
    tag_rows = _load_tag_metadata(model_path)
    tag_meta_count = len(tag_rows) if tag_rows else 0
    if tag_rows and len(tag_rows) == len(labels):
        labels = [row[0] for row in tag_rows]
        categories = [row[1] for row in tag_rows]

    bundle = {
        "model": model,
        "processor": processor,
        "labels": labels,
        "categories": categories,
        "device": resolved_device,
        "warn": warn,
        "torch": torch,
        "model_path": str(model_path),
        "tag_meta_loaded": bool(categories),
        "tag_meta_count": tag_meta_count,
    }
    _MODEL_CACHE[cache_key] = bundle
    return bundle


def _is_rating_tag(tag: str, category: Optional[int]) -> bool:
    if tag.lower().startswith("rating:"):
        return True
    if category is not None and category == CATEGORY_RATING:
        return True
    return False


def _format_tag(tag: str, replace_underscore: bool) -> str:
    return tag.replace("_", " ") if replace_underscore else tag


def _build_tags(
    probs,
    labels: List[str],
    categories: Optional[List[Optional[int]]],
    opts: TaggerOptions,
    stats: Optional[Dict[str, int]] = None,
) -> List[str]:
    general: List[Tuple[str, float]] = []
    characters: List[Tuple[str, float]] = []
    ratings: List[Tuple[str, float]] = []

    for idx, score in enumerate(probs):
        tag = labels[idx]
        category = categories[idx] if categories is not None and idx < len(categories) else None
        if _is_rating_tag(tag, category):
            ratings.append((tag, float(score)))
            continue
        if category == CATEGORY_CHARACTER:
            if opts.include_character and score >= opts.character_threshold:
                characters.append((tag, float(score)))
            continue
        if score >= opts.general_threshold:
            general.append((tag, float(score)))

    general.sort(key=lambda x: x[1], reverse=True)
    characters.sort(key=lambda x: x[1], reverse=True)

    tags = [t for t, _ in general] + [t for t, _ in characters]
    if opts.max_tags > 0 and len(tags) > opts.max_tags:
        tags = tags[: opts.max_tags]

    if opts.include_rating and ratings:
        rating_tag = max(ratings, key=lambda x: x[1])[0]
        tags = [rating_tag] + tags

    if stats is not None:
        stats["general"] = stats.get("general", 0) + len(general)
        stats["character"] = stats.get("character", 0) + len(characters)
        stats["rating"] = stats.get("rating", 0) + (1 if (opts.include_rating and ratings) else 0)
        if general:
            stats["images_with_general"] = stats.get("images_with_general", 0) + 1
        if characters:
            stats["images_with_character"] = stats.get("images_with_character", 0) + 1

    out: List[str] = []
    seen = set()
    for tag in tags:
        fmt = _format_tag(tag, opts.replace_underscore)
        if fmt not in seen:
            out.append(fmt)
            seen.add(fmt)
    return out


def _dedup_preserve(tags: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for tag in tags:
        if tag not in seen:
            out.append(tag)
            seen.add(tag)
    return out


def _read_tag_file(path: Path) -> Tuple[List[str], List[str], Optional[str]]:
    try:
        from services import normalizer

        tf = normalizer.parse_tag_file(path)
        return tf.main, tf.optional, tf.warning
    except Exception:
        try:
            text = path.read_text(encoding="utf-8")
            return split_tags(text), [], None
        except Exception:
            return [], [], None


def _format_tag_file(main: List[str], optional: List[str], warning: Optional[str]) -> str:
    try:
        from services import normalizer

        tf = normalizer.TagFile(path=Path(""), main=main, optional=optional, warning=warning)
        return normalizer.format_tag_file(tf)
    except Exception:
        return join_tags(main)


def run_tagger(opts: TaggerOptions) -> Tuple[bool, List[str]]:
    lines: List[str] = []

    if not opts.image_exts:
        opts.image_exts = list(DEFAULT_IMAGE_EXTS)
    if opts.write_mode not in {"overwrite", "append", "skip"}:
        opts.write_mode = "append"

    if not opts.dataset_path.exists() or not opts.dataset_path.is_dir():
        lines.append(f"Dataset folder not found: {opts.dataset_path}")
        return False, lines

    lines.append(f"Dataset: {opts.dataset_path}")
    lines.append(f"Model: {opts.model_id}")

    try:
        bundle = _load_model_bundle(opts.model_id, opts.device, opts.local_only)
    except Exception as exc:
        lines.append(f"Failed to load model: {exc}")
        return False, lines

    model = bundle["model"]
    processor = bundle["processor"]
    labels = bundle["labels"]
    categories = bundle["categories"]
    device = bundle["device"]
    torch = bundle["torch"]
    if bundle.get("model_path"):
        lines.append(f"Model path: {bundle['model_path']}")

    lines.append(f"Device: {device}")
    if bundle.get("warn"):
        lines.append(str(bundle["warn"]))
    if bundle.get("tag_meta_loaded"):
        lines.append(f"Tag categories: loaded ({bundle.get('tag_meta_count', 0)} tags).")
    else:
        lines.append("Tag categories: not loaded (missing tag CSV or length mismatch).")
    lines.append(f"Include character tags: {'on' if opts.include_character else 'off'}")
    lines.append(f"Include rating tags: {'on' if opts.include_rating else 'off'}")
    lines.append(f"Thresholds: general={opts.general_threshold}, character={opts.character_threshold}")

    paths = _iter_images(opts.dataset_path, opts.recursive, opts.image_exts)
    if opts.limit > 0:
        paths = paths[: opts.limit]

    if not paths:
        lines.append("No images found.")
        return False, lines

    skipped = 0
    if opts.write_mode == "skip":
        filtered = []
        for p in paths:
            txt_path = p.with_suffix(".txt")
            if txt_path.exists():
                try:
                    if txt_path.read_text(encoding="utf-8").strip():
                        skipped += 1
                        continue
                except Exception:
                    pass
            filtered.append(p)
        paths = filtered

    if not paths:
        lines.append(f"Skipped all files ({skipped}).")
        return True, lines

    processed = 0
    written = 0
    errors = 0
    samples: List[str] = []
    total_images = len(paths)
    lines.append(f"Total images: {total_images}")
    tag_stats: Dict[str, int] = {}

    try:
        from PIL import Image
    except Exception as exc:
        lines.append(f"Failed to load Pillow: {exc}")
        return False, lines

    for batch in _chunked(paths, opts.batch_size):
        images = []
        batch_paths = []
        for path in batch:
            try:
                with Image.open(path) as im:
                    images.append(im.convert("RGB"))
                batch_paths.append(path)
            except Exception as exc:
                errors += 1
                lines.append(f"[ERROR] {path.name}: {exc}")

        if not images:
            continue

        try:
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
        except Exception as exc:
            errors += len(batch_paths)
            lines.append(f"[ERROR] batch failed: {exc}")
            continue

        for path, row in zip(batch_paths, probs):
            tags = _build_tags(row, labels, categories, opts, tag_stats)
            if opts.skip_empty and not tags:
                processed += 1
                print(f"{processed} out of {total_images} images tagged.")
                continue

            if opts.preview_limit > 0 and len(samples) < opts.preview_limit:
                samples.append(f"{path.name}: {', '.join(tags)}")

            if opts.preview_only:
                processed += 1
                print(f"{processed} out of {total_images} images tagged.")
                continue

            txt_path = path.with_suffix(".txt")
            existing_main: List[str] = []
            existing_optional: List[str] = []
            existing_warning: Optional[str] = None
            if txt_path.exists():
                existing_main, existing_optional, existing_warning = _read_tag_file(txt_path)

            if opts.write_mode == "overwrite":
                merged_main = tags
            else:
                merged_main = _dedup_preserve(existing_main + tags)

            if merged_main or not opts.skip_empty:
                text = _format_tag_file(merged_main, existing_optional, existing_warning)
                txt_path.write_text(text, encoding="utf-8")
                written += 1

            processed += 1
            print(f"{processed} out of {total_images} images tagged.")

    if samples:
        lines.append("Sample tags:")
        lines.extend(samples)

    if tag_stats:
        lines.append(
            "Tag stats: "
            f"general={tag_stats.get('general', 0)}, "
            f"character={tag_stats.get('character', 0)}, "
            f"rating={tag_stats.get('rating', 0)}, "
            f"images_with_general={tag_stats.get('images_with_general', 0)}, "
            f"images_with_character={tag_stats.get('images_with_character', 0)}"
        )

    if opts.preview_only:
        lines.append(f"Preview done. Processed {processed} images (no files written).")
    else:
        lines.append(
            f"Done. Processed {processed} images, wrote {written} tag files, skipped {skipped}, errors {errors}."
        )

    return True, lines


def handle(form, ctx) -> Tuple[str, str]:
    active_tab = "offline_tagger"

    raw_folder = (form.get("folder") or "").strip()
    if not raw_folder:
        return active_tab, log_join(["Dataset folder is required."])

    dataset_path = readable_path(raw_folder)
    if not dataset_path.exists() or not dataset_path.is_dir():
        return active_tab, log_join([f"Dataset folder not found: {dataset_path}"])

    opts = TaggerOptions(
        dataset_path=dataset_path,
        recursive=_parse_bool(form.get("recursive")),
        image_exts=_parse_exts(form.get("exts") or ""),
        model_id=(form.get("model_id") or DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID,
        device=(form.get("device") or "auto").strip(),
        batch_size=max(1, _parse_int(form.get("batch_size"), 4)),
        general_threshold=_parse_float(form.get("general_threshold"), DEFAULT_GENERAL_THRESHOLD),
        character_threshold=_parse_float(form.get("character_threshold"), DEFAULT_CHARACTER_THRESHOLD),
        include_character=_parse_bool(form.get("include_character")),
        include_rating=_parse_bool(form.get("include_rating")),
        replace_underscore=_parse_bool(form.get("replace_underscore")),
        write_mode=(form.get("write_mode") or "append").strip().lower(),
        preview_only=_parse_bool(form.get("preview_only")),
        preview_limit=max(0, _parse_int(form.get("preview_limit"), 5)),
        limit=max(0, _parse_int(form.get("limit"), 0)),
        max_tags=max(0, _parse_int(form.get("max_tags"), 0)),
        skip_empty=_parse_bool(form.get("skip_empty")),
        local_only=_parse_bool(form.get("local_only")),
    )

    ok, lines = run_tagger(opts)
    return active_tab, log_join(lines)
