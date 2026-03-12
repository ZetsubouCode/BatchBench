from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from utils.io import readable_path
from utils.dataset import split_tags, join_tags
from utils.parse import (
    parse_bool,
    parse_exts,
    parse_float,
    parse_int,
    parse_line_list,
    parse_optional_int,
    parse_tag_list,
)
from utils.tool_result import build_tool_result


DEFAULT_MODEL_ID = "SmilingWolf/wd-swinv2-tagger-v3"
DEFAULT_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

DEFAULT_GENERAL_THRESHOLD = 0.35
DEFAULT_CHARACTER_THRESHOLD = 0.75
DEFAULT_THRESHOLD_MODE = "mcut"
DEFAULT_MIN_THRESHOLD_FLOOR = 0.2
DEFAULT_TAG_FOCUS_MODE = "all"

DEFAULT_MAX_GENERAL_TAGS = 0
DEFAULT_MAX_CHARACTER_TAGS = 0
DEFAULT_MAX_META_TAGS = 0

DEFAULT_MCUT_RELAX_GENERAL = 0.08
DEFAULT_MCUT_RELAX_CHARACTER = 0.02
DEFAULT_MCUT_RELAX_META = 0.05
DEFAULT_MCUT_MIN_GENERAL_TAGS = 8
DEFAULT_MCUT_MIN_CHARACTER_TAGS = 0
DEFAULT_MCUT_MIN_META_TAGS = 0

DEFAULT_BACKEND = "transformers"
DEFAULT_ENABLE_COLOR_SANITY = True
DEFAULT_COLOR_RATIO_THRESHOLD = 0.006
DEFAULT_COLOR_MIN_SATURATION = 0.20
DEFAULT_COLOR_MIN_VALUE = 0.15
DEFAULT_COLOR_KEEP_IF_SCORE_GE = 0.92
DEFAULT_COLOR_DOWNSCALE = 256
DEFAULT_DEBUG_COLOR_SANITY = False

DEFAULT_NON_CHARACTER_REGEX = [
    r"(?:^|[ _])(background|scenery|landscape|cityscape)(?:$|[ _])",
    r"(?:^|[ _])(indoors|outdoors|sky|cloud|sunset|sunrise|moon|star|night|day)(?:$|[ _])",
    r"(?:^|[ _])(room|bedroom|bathroom|kitchen|classroom|office|library|corridor|hallway)(?:$|[ _])",
    r"(?:^|[ _])(street|road|alley|bridge|sidewalk|building|window|door)(?:$|[ _])",
    r"(?:^|[ _])(forest|tree|grass|flower|mountain|river|lake|sea|ocean|beach)(?:$|[ _])",
    r"(?:^|[ _])(car|bus|train|airplane|ship|bicycle)(?:$|[ _])",
    r" background$",
]

TAGGER_POLICY = {
    "force_wd_bgr_fix": True,
    "model_id": DEFAULT_MODEL_ID,
    "device": "auto",
    "backend": DEFAULT_BACKEND,
    "use_amp": False,
    "batch_size": 4,
    "image_exts": list(DEFAULT_IMAGE_EXTS),
    "general_threshold": DEFAULT_GENERAL_THRESHOLD,
    "character_threshold": DEFAULT_CHARACTER_THRESHOLD,
    "threshold_mode": DEFAULT_THRESHOLD_MODE,
    "min_threshold_floor": DEFAULT_MIN_THRESHOLD_FLOOR,
    "mcut_relax_general": DEFAULT_MCUT_RELAX_GENERAL,
    "mcut_relax_character": DEFAULT_MCUT_RELAX_CHARACTER,
    "mcut_relax_meta": DEFAULT_MCUT_RELAX_META,
    "mcut_min_general_tags": DEFAULT_MCUT_MIN_GENERAL_TAGS,
    "mcut_min_character_tags": DEFAULT_MCUT_MIN_CHARACTER_TAGS,
    "mcut_min_meta_tags": DEFAULT_MCUT_MIN_META_TAGS,
    "tag_focus_mode": DEFAULT_TAG_FOCUS_MODE,
    "include_general": True,
    "include_character": True,
    "include_rating": False,
    "include_meta": False,
    "include_copyright": False,
    "include_artist": False,
    "replace_underscore": False,
    "write_mode": "append",
    "preview_only": False,
    "preview_limit": 20,
    "limit": 0,
    "max_tags": 0,
    "max_general_tags": DEFAULT_MAX_GENERAL_TAGS,
    "max_character_tags": DEFAULT_MAX_CHARACTER_TAGS,
    "max_meta_tags": DEFAULT_MAX_META_TAGS,
    "skip_empty": True,
    "local_only": False,
    "exclude_tags": "",
    "exclude_regex": "",
    "non_character_regex": list(DEFAULT_NON_CHARACTER_REGEX),
    "use_normalizer_remove_as_exclude": False,
    "trigger_tag": "",
    "general_category_id": None,
    "character_category_id": None,
    "rating_category_id": None,
    "normalizer_preset_type": "",
    "normalizer_preset_file": "",
    "dedupe": True,
    "sort_tags": True,
    "keep_existing_tags": True,
    "character_topk": 0,
    "newline_end": True,
    "strip_whitespace": True,
    "enable_color_sanity": DEFAULT_ENABLE_COLOR_SANITY,
    "color_ratio_threshold": DEFAULT_COLOR_RATIO_THRESHOLD,
    "color_min_saturation": DEFAULT_COLOR_MIN_SATURATION,
    "color_min_value": DEFAULT_COLOR_MIN_VALUE,
    "color_keep_if_score_ge": DEFAULT_COLOR_KEEP_IF_SCORE_GE,
    "color_downscale": DEFAULT_COLOR_DOWNSCALE,
    "debug_color_sanity": DEFAULT_DEBUG_COLOR_SANITY,
}

DEPRECATED_KEYS = {
    "device",
    "backend",
    "use_amp",
    "exts",
    "image_exts",
    "input_color_order",
    "enable_color_sanity",
    "color_ratio_threshold",
    "color_min_saturation",
    "color_min_value",
    "color_keep_if_score_ge",
    "color_downscale",
    "debug_color_sanity",
    "min_threshold_floor",
    "include_general",
    "include_meta",
    "include_copyright",
    "include_artist",
    "replace_underscore",
    "max_tags",
    "max_meta_tags",
    "skip_empty",
    "local_only",
    "exclude_regex",
    "use_normalizer_remove_as_exclude",
    "general_category_id",
    "character_category_id",
    "rating_category_id",
    "normalizer_preset_type",
    "normalizer_preset_file",
}

DEFAULT_CATEGORY_IDS = {
    "general": 0,
    "artist": 1,
    "copyright": 2,
    "character": 3,
    "meta": 4,
    "rating": 9,
}

RATING_TAG_HINTS = {
    "rating:safe",
    "rating:questionable",
    "rating:explicit",
    "rating:sensitive",
    "rating:general",
}

RATING_BARE_HINTS = {"safe", "questionable", "explicit", "sensitive"}

_COLOR_HUE_RANGES = {
    "red": [(0.0, 15.0), (350.0, 360.0)],
    "orange": [(15.0, 40.0)],
    "yellow": [(40.0, 70.0)],
    "green": [(70.0, 160.0)],
    "cyan": [(160.0, 200.0)],
    "blue": [(200.0, 260.0)],
    "purple": [(260.0, 290.0)],
    "pink": [(290.0, 350.0)],
}

_COLOR_SPECIAL = ("brown", "white", "black", "gray")
_COLOR_NAMES = tuple(list(_COLOR_HUE_RANGES.keys()) + list(_COLOR_SPECIAL))
_COLOR_ALIASES = {"grey": "gray"}
_COLOR_ATTR_SUFFIXES = ("hair", "eyes", "skin")

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
    threshold_mode: str
    min_threshold_floor: float
    mcut_relax_general: float
    mcut_relax_character: float
    mcut_relax_meta: float
    mcut_min_general_tags: int
    mcut_min_character_tags: int
    mcut_min_meta_tags: int
    tag_focus_mode: str
    include_general: bool
    include_character: bool
    include_rating: bool
    include_meta: bool
    include_copyright: bool
    include_artist: bool
    replace_underscore: bool
    write_mode: str
    preview_only: bool
    preview_limit: int
    limit: int
    max_tags: int
    max_general_tags: int
    max_character_tags: int
    max_meta_tags: int
    character_topk: int
    skip_empty: bool
    local_only: bool
    exclude_tags: List[str]
    exclude_regex: List[str]
    non_character_regex: List[str]
    use_normalizer_remove_as_exclude: bool
    backend: str
    use_amp: bool
    trigger_tag: str
    dedupe: bool
    sort_tags: bool
    keep_existing_tags: bool
    newline_end: bool
    strip_whitespace: bool
    force_wd_bgr_fix: bool
    general_category_id: Optional[int]
    character_category_id: Optional[int]
    rating_category_id: Optional[int]
    normalizer_preset_root: Optional[Path]
    normalizer_preset_type: str
    normalizer_preset_file: str
    enable_color_sanity: bool
    color_ratio_threshold: float
    color_min_saturation: float
    color_min_value: float
    color_keep_if_score_ge: float
    color_downscale: int
    debug_color_sanity: bool


@dataclass
class CategoryIds:
    general: Optional[int] = None
    character: Optional[int] = None
    rating: Optional[int] = None
    meta: Optional[int] = None
    copyright: Optional[int] = None
    artist: Optional[int] = None


def _parse_bool(val: Any) -> bool:
    return parse_bool(val, default=False)


def _parse_int(val: Any, default: int) -> int:
    return parse_int(val, default=default)


def _parse_float(val: Any, default: float) -> float:
    parsed = parse_float(val, default=default)
    return parsed if parsed is not None else default


def _parse_optional_int(val: Any) -> Optional[int]:
    return parse_optional_int(val)


def _parse_tag_list(raw: Any) -> List[str]:
    return parse_tag_list(raw, dedupe=False)


def _parse_regex_list(raw: Any) -> List[str]:
    return parse_line_list(raw, dedupe=False)


def _parse_exts(raw: Any) -> List[str]:
    return parse_exts(raw, default=DEFAULT_IMAGE_EXTS)


def _is_wd_family(model_id: str) -> bool:
    text = (model_id or "").strip().lower()
    if not text:
        return False
    if "wd" in text and "tagger" in text:
        return True
    for token in ("wd14", "wd-swinv2", "wd-v3", "wd-v4", "smilingwolf/wd"):
        if token in text:
            return True
    return False


def _find_deprecated_keys(form_opts: Dict[str, Any]) -> List[str]:
    return sorted({k for k in (form_opts or {}).keys() if k in DEPRECATED_KEYS})


def _effective_opts(form_opts: Dict[str, Any], policy: Dict[str, Any]) -> TaggerOptions:
    policy = policy or TAGGER_POLICY
    raw_input = (form_opts.get("input_dir") or form_opts.get("folder") or "").strip()
    dataset_path = readable_path(raw_input) if raw_input else Path(".")
    write_mode = (form_opts.get("write_mode") or policy.get("write_mode") or "append").strip().lower()
    if write_mode == "skip_if_exists":
        write_mode = "skip"
    if write_mode not in {"overwrite", "append", "skip"}:
        write_mode = "append"

    def _fallback(key: str, default_key: Optional[str] = None):
        if key in form_opts and form_opts.get(key) not in (None, ""):
            return form_opts.get(key)
        if default_key and default_key in form_opts and form_opts.get(default_key) not in (None, ""):
            return form_opts.get(default_key)
        return policy.get(key)

    general_threshold = _parse_float(
        _fallback("min_general", "general_threshold"), policy.get("general_threshold", DEFAULT_GENERAL_THRESHOLD)
    )
    character_threshold = _parse_float(
        _fallback("min_character", "character_threshold"),
        policy.get("character_threshold", DEFAULT_CHARACTER_THRESHOLD),
    )
    threshold_mode = (form_opts.get("threshold_mode") or policy.get("threshold_mode") or DEFAULT_THRESHOLD_MODE).strip().lower()
    if threshold_mode not in {"fixed", "mcut"}:
        threshold_mode = DEFAULT_THRESHOLD_MODE
    tag_focus_mode = (form_opts.get("tag_focus_mode") or policy.get("tag_focus_mode") or DEFAULT_TAG_FOCUS_MODE).strip().lower()
    if tag_focus_mode not in {"all", "character", "non_character"}:
        tag_focus_mode = DEFAULT_TAG_FOCUS_MODE

    batch_size = max(1, _parse_int(_fallback("batch_size"), int(policy.get("batch_size", 4))))
    preview_limit = max(0, _parse_int(_fallback("preview_limit"), int(policy.get("preview_limit", 20))))
    limit = max(0, _parse_int(_fallback("limit"), int(policy.get("limit", 0))))

    include_general = bool(policy.get("include_general", True))
    include_character = _parse_bool(_fallback("include_character")) if "include_character" in form_opts else bool(
        policy.get("include_character", True)
    )
    if tag_focus_mode == "character":
        include_general = True
        include_character = True
    elif tag_focus_mode == "non_character":
        include_general = True
        include_character = False
    include_rating = _parse_bool(_fallback("include_rating")) if "include_rating" in form_opts else bool(
        policy.get("include_rating", False)
    )

    dedupe = _parse_bool(_fallback("dedupe")) if "dedupe" in form_opts else bool(policy.get("dedupe", True))
    sort_tags = _parse_bool(_fallback("sort_tags")) if "sort_tags" in form_opts else bool(policy.get("sort_tags", True))
    keep_existing_tags = (
        _parse_bool(_fallback("keep_existing_tags"))
        if "keep_existing_tags" in form_opts
        else bool(policy.get("keep_existing_tags", True))
    )
    if write_mode != "append":
        keep_existing_tags = False

    return TaggerOptions(
        dataset_path=dataset_path,
        recursive=_parse_bool(_fallback("recursive")) if "recursive" in form_opts else bool(policy.get("recursive", False)),
        image_exts=_parse_exts(policy.get("image_exts") or DEFAULT_IMAGE_EXTS),
        model_id=(form_opts.get("model_id") or policy.get("model_id") or DEFAULT_MODEL_ID).strip()
        or DEFAULT_MODEL_ID,
        device=(policy.get("device") or "auto").strip(),
        batch_size=batch_size,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        threshold_mode=threshold_mode,
        min_threshold_floor=float(policy.get("min_threshold_floor", DEFAULT_MIN_THRESHOLD_FLOOR)),
        mcut_relax_general=max(
            0.0,
            min(
                _parse_float(_fallback("mcut_relax_general"), float(policy.get("mcut_relax_general", DEFAULT_MCUT_RELAX_GENERAL))),
                1.0,
            ),
        ),
        mcut_relax_character=max(
            0.0,
            min(
                _parse_float(
                    _fallback("mcut_relax_character"),
                    float(policy.get("mcut_relax_character", DEFAULT_MCUT_RELAX_CHARACTER)),
                ),
                1.0,
            ),
        ),
        mcut_relax_meta=max(
            0.0,
            min(
                _parse_float(_fallback("mcut_relax_meta"), float(policy.get("mcut_relax_meta", DEFAULT_MCUT_RELAX_META))),
                1.0,
            ),
        ),
        mcut_min_general_tags=max(
            0,
            _parse_int(_fallback("mcut_min_general_tags"), int(policy.get("mcut_min_general_tags", DEFAULT_MCUT_MIN_GENERAL_TAGS))),
        ),
        mcut_min_character_tags=max(
            0,
            _parse_int(
                _fallback("mcut_min_character_tags"),
                int(policy.get("mcut_min_character_tags", DEFAULT_MCUT_MIN_CHARACTER_TAGS)),
            ),
        ),
        mcut_min_meta_tags=max(
            0,
            _parse_int(_fallback("mcut_min_meta_tags"), int(policy.get("mcut_min_meta_tags", DEFAULT_MCUT_MIN_META_TAGS))),
        ),
        tag_focus_mode=tag_focus_mode,
        include_general=include_general,
        include_character=include_character,
        include_rating=include_rating,
        include_meta=bool(policy.get("include_meta", False)),
        include_copyright=bool(policy.get("include_copyright", False)),
        include_artist=bool(policy.get("include_artist", False)),
        replace_underscore=bool(policy.get("replace_underscore", False)),
        write_mode=write_mode,
        preview_only=_parse_bool(_fallback("preview_only"))
        if "preview_only" in form_opts
        else bool(policy.get("preview_only", False)),
        preview_limit=preview_limit,
        limit=limit,
        max_tags=max(0, _parse_int(policy.get("max_tags", 0), 0)),
        max_general_tags=max(0, _parse_int(_fallback("max_general_tags"), int(policy.get("max_general_tags", 0)))),
        max_character_tags=max(
            0, _parse_int(_fallback("max_character_tags"), int(policy.get("max_character_tags", 0)))
        ),
        max_meta_tags=max(0, _parse_int(policy.get("max_meta_tags", 0), 0)),
        character_topk=max(0, _parse_int(_fallback("character_topk"), int(policy.get("character_topk", 0)))),
        skip_empty=bool(policy.get("skip_empty", True)),
        local_only=bool(policy.get("local_only", False)),
        exclude_tags=_parse_tag_list(_fallback("exclude_tags") or policy.get("exclude_tags") or ""),
        exclude_regex=_parse_regex_list(policy.get("exclude_regex") or []),
        non_character_regex=_parse_regex_list(
            _fallback("non_character_regex") or policy.get("non_character_regex") or DEFAULT_NON_CHARACTER_REGEX
        ),
        use_normalizer_remove_as_exclude=bool(policy.get("use_normalizer_remove_as_exclude", False)),
        backend=(policy.get("backend") or DEFAULT_BACKEND).strip().lower(),
        use_amp=bool(policy.get("use_amp", False)),
        trigger_tag=(form_opts.get("trigger_tag") or policy.get("trigger_tag") or "").strip(),
        dedupe=dedupe,
        sort_tags=sort_tags,
        keep_existing_tags=keep_existing_tags,
        newline_end=bool(policy.get("newline_end", True)),
        strip_whitespace=bool(policy.get("strip_whitespace", True)),
        force_wd_bgr_fix=bool(policy.get("force_wd_bgr_fix", True)),
        general_category_id=None,
        character_category_id=None,
        rating_category_id=None,
        normalizer_preset_root=readable_path(str(policy.get("normalizer_preset_root")))
        if policy.get("normalizer_preset_root")
        else None,
        normalizer_preset_type=str(policy.get("normalizer_preset_type") or ""),
        normalizer_preset_file=str(policy.get("normalizer_preset_file") or ""),
        enable_color_sanity=bool(policy.get("enable_color_sanity", DEFAULT_ENABLE_COLOR_SANITY)),
        color_ratio_threshold=float(policy.get("color_ratio_threshold", DEFAULT_COLOR_RATIO_THRESHOLD)),
        color_min_saturation=float(policy.get("color_min_saturation", DEFAULT_COLOR_MIN_SATURATION)),
        color_min_value=float(policy.get("color_min_value", DEFAULT_COLOR_MIN_VALUE)),
        color_keep_if_score_ge=float(policy.get("color_keep_if_score_ge", DEFAULT_COLOR_KEEP_IF_SCORE_GE)),
        color_downscale=max(16, int(policy.get("color_downscale", DEFAULT_COLOR_DOWNSCALE))),
        debug_color_sanity=bool(policy.get("debug_color_sanity", DEFAULT_DEBUG_COLOR_SANITY)),
    )


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


def _find_onnx_model(model_path: Path) -> Optional[Path]:
    candidates = sorted(model_path.rglob("*.onnx"))
    if not candidates:
        return None
    for name in ["model.onnx", "model_fp16.onnx", "model_fp32.onnx"]:
        for cand in candidates:
            if cand.name.lower() == name:
                return cand
    return candidates[0]


def _load_model_bundle(model_id: str, device: str, local_only: bool, backend: str):
    backend = (backend or DEFAULT_BACKEND).strip().lower()
    cache_key = (model_id, device, backend)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
    except Exception as exc:
        raise RuntimeError(
            "Missing deps for offline tagger. Install torch and transformers first."
        ) from exc

    model_path = _ensure_model_local(model_id, local_only)
    processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)

    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    num_labels = int(getattr(config, "num_labels", 0)) or len(getattr(config, "id2label", {}))
    id2label = getattr(config, "id2label", {}) or {}
    labels = [str(id2label.get(i) or id2label.get(str(i)) or f"tag_{i}") for i in range(num_labels)]

    categories: Optional[List[Optional[int]]] = None
    tag_rows = _load_tag_metadata(model_path)
    tag_meta_count = len(tag_rows) if tag_rows else 0
    if tag_rows and len(tag_rows) == len(labels):
        labels = [row[0] for row in tag_rows]
        categories = [row[1] for row in tag_rows]

    warn: List[str] = []
    torch = None
    model = None
    onnx_session = None
    onnx_input = None
    onnx_path = None
    resolved_device = device
    provider = None

    if backend == "onnx":
        try:
            import onnxruntime as ort
        except Exception:
            warn.append("onnxruntime not available; falling back to transformers.")
            backend = "transformers"
        else:
            onnx_path = _find_onnx_model(model_path)
            if not onnx_path:
                warn.append("ONNX model file not found; falling back to transformers.")
                backend = "transformers"
            else:
                providers = ort.get_available_providers()
                if str(device).lower().startswith("cuda") and "CUDAExecutionProvider" in providers:
                    provider = "CUDAExecutionProvider"
                    provider_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    resolved_device = "cuda"
                else:
                    provider = "CPUExecutionProvider"
                    provider_list = ["CPUExecutionProvider"]
                    if str(device).lower().startswith("cuda"):
                        warn.append("CUDA provider not available for ONNX; using CPU.")
                    resolved_device = "cpu"
                onnx_session = ort.InferenceSession(str(onnx_path), providers=provider_list)
                onnx_input = onnx_session.get_inputs()[0].name if onnx_session.get_inputs() else None

    if backend != "onnx":
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "Missing torch dependency; cannot run transformers backend."
            ) from exc
        resolved_device, device_warn = _resolve_device(device, torch)
        if device_warn:
            warn.append(device_warn)
        model = AutoModelForImageClassification.from_pretrained(
            str(model_path),
            local_files_only=True,
            use_safetensors=True,
        )
        model.eval()
        model.to(resolved_device)
        provider = None

    actual_key = (model_id, resolved_device, backend)
    if actual_key in _MODEL_CACHE:
        return _MODEL_CACHE[actual_key]

    bundle = {
        "backend": backend,
        "model": model,
        "onnx_session": onnx_session,
        "onnx_input": onnx_input,
        "onnx_path": str(onnx_path) if onnx_path else None,
        "processor": processor,
        "labels": labels,
        "categories": categories,
        "device": resolved_device,
        "warn": warn,
        "torch": torch,
        "provider": provider,
        "model_path": str(model_path),
        "tag_meta_loaded": bool(categories),
        "tag_meta_count": tag_meta_count,
    }
    _MODEL_CACHE[actual_key] = bundle
    return bundle


def _guess_rating_category_id(
    labels: List[str], categories: Optional[List[Optional[int]]]
) -> Optional[int]:
    if not categories:
        return None
    scores: Dict[int, int] = {}
    for label, cat in zip(labels, categories):
        if cat is None:
            continue
        name = label.strip().lower()
        if name in RATING_TAG_HINTS:
            scores[cat] = scores.get(cat, 0) + 3
        elif name.startswith("rating:"):
            scores[cat] = scores.get(cat, 0) + 2
        elif name in RATING_BARE_HINTS:
            scores[cat] = scores.get(cat, 0) + 1
    if not scores:
        return None
    return max(scores.items(), key=lambda kv: kv[1])[0]


def resolve_category_ids(
    labels: List[str],
    categories: Optional[List[Optional[int]]],
    overrides: TaggerOptions,
) -> Tuple[CategoryIds, List[str]]:
    warnings: List[str] = []
    if not categories:
        warnings.append("Tag categories missing; category filtering will be limited.")
        return CategoryIds(), warnings

    available = {c for c in categories if c is not None}

    def _resolve(name: str, override: Optional[int], default_id: Optional[int]) -> Optional[int]:
        if override is not None:
            if override in available:
                return override
            warnings.append(f"{name} category override {override} not present in tag metadata.")
            return None
        if default_id is not None and default_id in available:
            return default_id
        warnings.append(f"{name} category id not resolved from metadata.")
        return None

    rating_guess = _guess_rating_category_id(labels, categories)
    rating_override = overrides.rating_category_id
    rating_id = None
    if rating_override is not None:
        if rating_override in available:
            rating_id = rating_override
        else:
            warnings.append(f"Rating category override {rating_override} not present in tag metadata.")
    elif rating_guess is not None:
        rating_id = rating_guess
    elif DEFAULT_CATEGORY_IDS["rating"] in available:
        rating_id = DEFAULT_CATEGORY_IDS["rating"]
        warnings.append("Rating category id not detected; using default 9.")
    else:
        warnings.append("Rating category id not detected; rating tags may be skipped.")

    return CategoryIds(
        general=_resolve("General", overrides.general_category_id, DEFAULT_CATEGORY_IDS.get("general")),
        character=_resolve("Character", overrides.character_category_id, DEFAULT_CATEGORY_IDS.get("character")),
        rating=rating_id,
        meta=_resolve("Meta", None, DEFAULT_CATEGORY_IDS.get("meta")),
        copyright=_resolve("Copyright", None, DEFAULT_CATEGORY_IDS.get("copyright")),
        artist=_resolve("Artist", None, DEFAULT_CATEGORY_IDS.get("artist")),
    ), warnings


def _is_rating_tag(tag: str, category: Optional[int], rating_category_id: Optional[int]) -> bool:
    if tag.lower().startswith("rating:"):
        return True
    if category is not None and rating_category_id is not None and category == rating_category_id:
        return True
    return False


_WORD_TAG_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _format_tag(tag: str, replace_underscore: bool) -> str:
    if not replace_underscore:
        return tag
    if not _WORD_TAG_RE.match(tag):
        return tag
    return tag.replace("_", " ")


def _swap_rgb_bgr(im):
    import numpy as np
    from PIL import Image

    arr = np.array(im)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[..., ::-1]
    return Image.fromarray(arr, mode="RGB")


def _normalize_tag_for_color(tag: str) -> str:
    return " ".join(tag.strip().lower().replace("_", " ").split())


def _is_color_attribute_tag(tag: str) -> Optional[str]:
    text = _normalize_tag_for_color(tag)
    for suffix in _COLOR_ATTR_SUFFIXES:
        if not text.endswith(f" {suffix}"):
            continue
        color = text[: -len(suffix)].strip()
        color = _COLOR_ALIASES.get(color, color)
        if color in _COLOR_NAMES:
            return color
    return None


def _estimate_color_presence(im, opts: TaggerOptions) -> Dict[str, float]:
    import numpy as np
    from PIL import Image

    max_side = max(1, int(opts.color_downscale))
    if max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side), Image.BILINEAR)
    hsv = im.convert("HSV")
    arr = np.asarray(hsv).astype(np.float32)
    if arr.size == 0:
        return {name: 0.0 for name in _COLOR_NAMES}

    hue = arr[..., 0] * (360.0 / 255.0)
    sat = arr[..., 1] / 255.0
    val = arr[..., 2] / 255.0
    total = float(hue.size) if hue.size else 1.0

    min_sat = max(0.0, min(float(opts.color_min_saturation), 1.0))
    min_val = max(0.0, min(float(opts.color_min_value), 1.0))
    valid = (sat >= min_sat) & (val >= min_val)

    counts: Dict[str, int] = {name: 0 for name in _COLOR_NAMES}

    for color, ranges in _COLOR_HUE_RANGES.items():
        mask = np.zeros(hue.shape, dtype=bool)
        for lo, hi in ranges:
            mask |= (hue >= lo) & (hue < hi)
        counts[color] = int(np.count_nonzero(valid & mask))

    white_mask = (sat <= 0.15) & (val >= 0.85)
    black_mask = val <= 0.1
    gray_mask = (sat <= 0.2) & (val > 0.1) & (val < 0.85)
    brown_mask = valid & (val < 0.6) & (hue >= 15.0) & (hue < 50.0)

    counts["white"] = int(np.count_nonzero(white_mask))
    counts["black"] = int(np.count_nonzero(black_mask))
    counts["gray"] = int(np.count_nonzero(gray_mask))
    counts["brown"] = int(np.count_nonzero(brown_mask))

    return {name: counts[name] / total for name in _COLOR_NAMES}


def _compile_regex(patterns: List[str]) -> List[re.Pattern]:
    out: List[re.Pattern] = []
    for pat in patterns or []:
        try:
            out.append(re.compile(pat, flags=re.IGNORECASE))
        except re.error:
            continue
    return out


def _matches_any(tag: str, patterns: List[re.Pattern]) -> bool:
    return any(pat.search(tag) for pat in patterns)


def _mcut_threshold(scores: List[float], floor: float, relax: float = 0.0, min_tags: int = 0) -> float:
    if not scores:
        return 1.1
    floor = max(0.0, min(float(floor), 1.0))
    relax = max(0.0, min(float(relax), 1.0))
    min_tags = max(0, int(min_tags))
    if len(scores) == 1:
        return max(scores[0] - relax, floor)
    sorted_scores = sorted(scores, reverse=True)
    gaps = [sorted_scores[i] - sorted_scores[i + 1] for i in range(len(sorted_scores) - 1)]
    max_idx = max(range(len(gaps)), key=lambda idx: gaps[idx])
    threshold = (sorted_scores[max_idx] + sorted_scores[max_idx + 1]) / 2.0
    threshold = threshold - relax
    if min_tags > 0:
        pivot_idx = min(min_tags - 1, len(sorted_scores) - 1)
        threshold = min(threshold, sorted_scores[pivot_idx])
    return max(threshold, floor)


def _split_general_focus(
    general: List[Tuple[str, float]],
    non_character_patterns: List[re.Pattern],
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    if not general:
        return [], []
    if not non_character_patterns:
        return list(general), []
    subject_general: List[Tuple[str, float]] = []
    non_character_general: List[Tuple[str, float]] = []
    for tag, score in general:
        norm = _normalize_tag_for_color(tag)
        if _matches_any(norm, non_character_patterns):
            non_character_general.append((tag, score))
        else:
            subject_general.append((tag, score))
    return subject_general, non_character_general


def _apply_excludes(tags: List[str], exclude_tags: set, exclude_regex: List[re.Pattern]) -> List[str]:
    if not exclude_tags and not exclude_regex:
        return tags
    out: List[str] = []
    for tag in tags:
        if tag in exclude_tags:
            continue
        if exclude_regex and _matches_any(tag, exclude_regex):
            continue
        out.append(tag)
    return out


def _apply_trigger_tag(tags: List[str], trigger_tag: str) -> List[str]:
    trigger = (trigger_tag or "").strip()
    if not trigger:
        return tags
    out = [trigger]
    for tag in tags:
        if tag != trigger:
            out.append(tag)
    return out


def _build_tags(
    probs,
    labels: List[str],
    categories: Optional[List[Optional[int]]],
    opts: TaggerOptions,
    category_ids: CategoryIds,
    exclude_tags: set,
    exclude_regex: List[re.Pattern],
    non_character_regex: Optional[List[re.Pattern]] = None,
    stats: Optional[Dict[str, int]] = None,
    color_presence: Optional[Dict[str, float]] = None,
    color_debug: Optional[List[str]] = None,
) -> List[str]:
    general: List[Tuple[str, float]] = []
    characters: List[Tuple[str, float]] = []
    meta: List[Tuple[str, float]] = []
    artists: List[Tuple[str, float]] = []
    copyrights: List[Tuple[str, float]] = []
    ratings: List[Tuple[str, float]] = []
    unknown: List[Tuple[str, float]] = []

    categories_present = bool(categories)

    use_color_sanity = bool(opts.enable_color_sanity and color_presence)
    ratio_threshold = max(0.0, min(float(opts.color_ratio_threshold), 1.0))
    keep_threshold = max(0.0, min(float(opts.color_keep_if_score_ge), 1.0))

    for idx, score in enumerate(probs):
        tag = labels[idx]
        category = categories[idx] if categories is not None and idx < len(categories) else None
        score_f = float(score)
        if use_color_sanity:
            color = _is_color_attribute_tag(tag)
            if color:
                presence = float(color_presence.get(color, 0.0)) if color_presence else 0.0
                if presence < ratio_threshold and score_f < keep_threshold:
                    if color_debug is not None:
                        color_debug.append(
                            f"{tag} (score={score_f:.3f}, presence={presence:.4f})"
                        )
                    continue
        if _is_rating_tag(tag, category, category_ids.rating):
            ratings.append((tag, score_f))
            continue
        if not categories_present:
            general.append((tag, score_f))
            continue
        if category is not None and category_ids.character is not None and category == category_ids.character:
            characters.append((tag, score_f))
        elif category is not None and category_ids.general is not None and category == category_ids.general:
            general.append((tag, score_f))
        elif category is not None and category_ids.meta is not None and category == category_ids.meta:
            meta.append((tag, score_f))
        elif category is not None and category_ids.artist is not None and category == category_ids.artist:
            artists.append((tag, score_f))
        elif category is not None and category_ids.copyright is not None and category == category_ids.copyright:
            copyrights.append((tag, score_f))
        else:
            unknown.append((tag, score_f))

    mode = (opts.threshold_mode or DEFAULT_THRESHOLD_MODE).strip().lower()
    floor = max(0.0, min(opts.min_threshold_floor, 1.0))
    use_mcut = mode == "mcut"
    focus_mode = (opts.tag_focus_mode or DEFAULT_TAG_FOCUS_MODE).strip().lower()
    if focus_mode not in {"all", "character", "non_character"}:
        focus_mode = DEFAULT_TAG_FOCUS_MODE

    def _threshold(items: List[Tuple[str, float]], fallback: float, relax: float, min_tags: int) -> float:
        if not use_mcut:
            return fallback
        scores = [score for _, score in items]
        return _mcut_threshold(scores, floor, relax=relax, min_tags=min_tags)

    general_thr = _threshold(
        general,
        opts.general_threshold,
        opts.mcut_relax_general,
        opts.mcut_min_general_tags,
    )
    character_thr = _threshold(
        characters,
        opts.character_threshold,
        opts.mcut_relax_character,
        opts.mcut_min_character_tags,
    )
    meta_thr = _threshold(
        meta + artists + copyrights,
        opts.general_threshold,
        opts.mcut_relax_meta,
        opts.mcut_min_meta_tags,
    )

    general = [item for item in general if item[1] >= general_thr]
    characters = [item for item in characters if item[1] >= character_thr]
    meta = [item for item in meta if item[1] >= meta_thr]
    artists = [item for item in artists if item[1] >= meta_thr]
    copyrights = [item for item in copyrights if item[1] >= meta_thr]

    general.sort(key=lambda x: x[1], reverse=True)
    characters.sort(key=lambda x: x[1], reverse=True)
    meta.sort(key=lambda x: x[1], reverse=True)
    artists.sort(key=lambda x: x[1], reverse=True)
    copyrights.sort(key=lambda x: x[1], reverse=True)

    if opts.character_topk > 0 and len(characters) > opts.character_topk:
        characters = characters[: opts.character_topk]

    if opts.max_general_tags > 0 and len(general) > opts.max_general_tags:
        general = general[: opts.max_general_tags]
    if opts.max_character_tags > 0 and len(characters) > opts.max_character_tags:
        characters = characters[: opts.max_character_tags]

    split_patterns = non_character_regex
    if split_patterns is None:
        split_patterns = _compile_regex(opts.non_character_regex)
    subject_general, non_character_general = _split_general_focus(general, split_patterns)

    meta_bucket: List[Tuple[str, float]] = []
    if opts.include_meta:
        meta_bucket += meta
    if opts.include_artist:
        meta_bucket += artists
    if opts.include_copyright:
        meta_bucket += copyrights
    meta_bucket.sort(key=lambda x: x[1], reverse=True)
    if opts.max_meta_tags > 0 and len(meta_bucket) > opts.max_meta_tags:
        meta_bucket = meta_bucket[: opts.max_meta_tags]

    tags: List[str] = []
    emitted_general: List[Tuple[str, float]] = []
    emitted_character: List[Tuple[str, float]] = []
    emitted_meta: List[Tuple[str, float]] = []
    if focus_mode == "character":
        if opts.include_general:
            emitted_general = subject_general
            tags.extend([t for t, _ in emitted_general])
        if opts.include_character:
            emitted_character = characters
            tags.extend([t for t, _ in emitted_character])
    elif focus_mode == "non_character":
        if opts.include_general:
            emitted_general = non_character_general
            tags.extend([t for t, _ in emitted_general])
        emitted_meta = meta_bucket
        tags.extend([t for t, _ in emitted_meta])
    else:
        if opts.include_general:
            emitted_general = general
            tags.extend([t for t, _ in emitted_general])
        if opts.include_character:
            emitted_character = characters
            tags.extend([t for t, _ in emitted_character])
        emitted_meta = meta_bucket
        tags.extend([t for t, _ in emitted_meta])

    if not categories_present and opts.include_general:
        unknown.sort(key=lambda x: x[1], reverse=True)
        tags.extend([t for t, _ in unknown])

    if opts.max_tags > 0 and len(tags) > opts.max_tags:
        tags = tags[: opts.max_tags]

    if opts.include_rating and ratings:
        rating_tag = max(ratings, key=lambda x: x[1])[0]
        tags = [rating_tag] + tags

    if stats is not None:
        stats["general"] = stats.get("general", 0) + len(emitted_general)
        stats["character"] = stats.get("character", 0) + len(emitted_character)
        stats["meta"] = stats.get("meta", 0) + len(emitted_meta)
        stats["subject_general"] = stats.get("subject_general", 0) + len(subject_general)
        stats["non_character_general"] = stats.get("non_character_general", 0) + len(non_character_general)
        stats["rating"] = stats.get("rating", 0) + (1 if (opts.include_rating and ratings) else 0)
        if emitted_general:
            stats["images_with_general"] = stats.get("images_with_general", 0) + 1
        if emitted_character:
            stats["images_with_character"] = stats.get("images_with_character", 0) + 1

    out: List[str] = []
    seen = set()
    for tag in tags:
        fmt = _format_tag(tag, opts.replace_underscore)
        if fmt not in seen:
            out.append(fmt)
            seen.add(fmt)

    out = _apply_excludes(out, exclude_tags, exclude_regex)
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


def _apply_output_formatting(text: str, newline_end: bool, strip_whitespace: bool) -> str:
    if strip_whitespace:
        lines = []
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            lines.append(line.strip() if line else "")
        text = "\n".join(lines)
    if newline_end:
        if not text.endswith("\n"):
            text += "\n"
    else:
        text = text.rstrip("\n")
    return text


def _format_tag_file(
    main: List[str],
    optional: List[str],
    warning: Optional[str],
    newline_end: bool,
    strip_whitespace: bool,
) -> str:
    try:
        from services import normalizer

        tf = normalizer.TagFile(path=Path(""), main=main, optional=optional, warning=warning)
        text = normalizer.format_tag_file(tf)
    except Exception:
        text = join_tags(main)
    return _apply_output_formatting(text, newline_end, strip_whitespace)


def _load_normalizer_excludes(
    preset_root: Optional[Path],
    preset_type: str,
    preset_file: str,
) -> Tuple[set, List[str], Optional[str]]:
    if not preset_root or not preset_file:
        return set(), [], "Normalizer preset not provided."
    try:
        from services import normalizer

        preset = normalizer.load_preset(preset_root, preset_type, preset_file)
    except Exception as exc:
        return set(), [], f"Failed to load normalizer preset: {exc}"
    rules = preset.get("rules", {})
    remove_tags = set(rules.get("remove_tags") or [])
    remove_regex = list(rules.get("remove_regex") or [])
    return remove_tags, remove_regex, None


def run_tagger(
    opts: TaggerOptions, deprecated_keys: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    lines: List[str] = []
    if deprecated_keys:
        lines.append(f"Ignored deprecated options: {', '.join(deprecated_keys)}")

    if not opts.image_exts:
        opts.image_exts = list(DEFAULT_IMAGE_EXTS)
    if opts.write_mode == "skip_if_exists":
        opts.write_mode = "skip"
    if opts.write_mode not in {"overwrite", "append", "skip"}:
        opts.write_mode = "append"
    if opts.write_mode != "append":
        opts.keep_existing_tags = False
    if (opts.threshold_mode or "").strip().lower() not in {"fixed", "mcut"}:
        opts.threshold_mode = DEFAULT_THRESHOLD_MODE
    if (opts.tag_focus_mode or "").strip().lower() not in {"all", "character", "non_character"}:
        opts.tag_focus_mode = DEFAULT_TAG_FOCUS_MODE
    if (opts.backend or "").strip().lower() not in {"transformers", "onnx"}:
        opts.backend = DEFAULT_BACKEND
    opts.min_threshold_floor = max(0.0, min(float(opts.min_threshold_floor), 1.0))
    opts.mcut_relax_general = max(0.0, min(float(opts.mcut_relax_general), 1.0))
    opts.mcut_relax_character = max(0.0, min(float(opts.mcut_relax_character), 1.0))
    opts.mcut_relax_meta = max(0.0, min(float(opts.mcut_relax_meta), 1.0))
    opts.mcut_min_general_tags = max(0, int(opts.mcut_min_general_tags))
    opts.mcut_min_character_tags = max(0, int(opts.mcut_min_character_tags))
    opts.mcut_min_meta_tags = max(0, int(opts.mcut_min_meta_tags))
    opts.trigger_tag = (opts.trigger_tag or "").strip()
    opts.color_ratio_threshold = max(0.0, min(float(opts.color_ratio_threshold), 1.0))
    opts.color_min_saturation = max(0.0, min(float(opts.color_min_saturation), 1.0))
    opts.color_min_value = max(0.0, min(float(opts.color_min_value), 1.0))
    opts.color_keep_if_score_ge = max(0.0, min(float(opts.color_keep_if_score_ge), 1.0))
    opts.color_downscale = max(16, int(opts.color_downscale or DEFAULT_COLOR_DOWNSCALE))

    if not opts.dataset_path.exists() or not opts.dataset_path.is_dir():
        lines.append(f"Dataset folder not found: {opts.dataset_path}")
        return False, lines

    lines.append(f"Dataset: {opts.dataset_path}")
    lines.append(f"Model: {opts.model_id}")
    wd_fix = bool(opts.force_wd_bgr_fix) and _is_wd_family(opts.model_id)
    lines.append(f"WD color fix {'ON (RGB->BGR)' if wd_fix else 'OFF'}")
    lines.append(
        f"Threshold mode: {opts.threshold_mode} "
        f"(general={opts.general_threshold}, character={opts.character_threshold})"
    )
    if opts.threshold_mode == "mcut":
        lines.append(
            "MCUT tuning: "
            f"relax(g/c/m)={opts.mcut_relax_general}/{opts.mcut_relax_character}/{opts.mcut_relax_meta}, "
            f"min_tags(g/c/m)={opts.mcut_min_general_tags}/{opts.mcut_min_character_tags}/{opts.mcut_min_meta_tags}"
        )
    lines.append(
        f"Output mode: {'skip_if_exists' if opts.write_mode == 'skip' else opts.write_mode}"
    )
    if opts.max_general_tags > 0 or opts.max_character_tags > 0:
        lines.append(
            f"Tag caps: general={opts.max_general_tags or 'unlimited'}, "
            f"character={opts.max_character_tags or 'unlimited'}"
        )
    if opts.character_topk > 0:
        lines.append(f"Character top-k: {opts.character_topk}")

    try:
        bundle = _load_model_bundle(opts.model_id, opts.device, opts.local_only, opts.backend)
    except Exception as exc:
        lines.append(f"Failed to load model: {exc}")
        return False, lines

    model = bundle["model"]
    backend = bundle.get("backend")
    onnx_session = bundle.get("onnx_session")
    onnx_input = bundle.get("onnx_input")
    processor = bundle["processor"]
    labels = bundle["labels"]
    categories = bundle["categories"]
    device = bundle["device"]
    torch = bundle["torch"]
    if bundle.get("model_path"):
        lines.append(f"Model path: {bundle['model_path']}")

    lines.append(f"Backend: {bundle.get('backend')}")
    lines.append(f"Device: {device}")
    if bundle.get("provider"):
        lines.append(f"ONNX provider: {bundle.get('provider')}")
    if bundle.get("backend") == "transformers":
        lines.append(f"AMP: {'on' if opts.use_amp else 'off'}")
    for warn in bundle.get("warn", []) or []:
        if warn:
            lines.append(str(warn))
    if bundle.get("tag_meta_loaded"):
        lines.append(f"Tag categories: loaded ({bundle.get('tag_meta_count', 0)} tags).")
    else:
        lines.append("Tag categories: not loaded (missing tag CSV or length mismatch).")
    category_ids, category_warnings = resolve_category_ids(labels, categories, opts)
    if category_warnings:
        for warn in category_warnings:
            lines.append(f"[WARN] {warn}")
    lines.append(
        "Category map: "
        f"general={category_ids.general}, "
        f"character={category_ids.character}, "
        f"rating={category_ids.rating}, "
        f"meta={category_ids.meta}, "
        f"artist={category_ids.artist}, "
        f"copyright={category_ids.copyright}"
    )
    lines.append(f"Include general tags: {'on' if opts.include_general else 'off'}")
    lines.append(f"Include character tags: {'on' if opts.include_character else 'off'}")
    lines.append(f"Include rating tags: {'on' if opts.include_rating else 'off'}")
    lines.append(f"Tag focus mode: {opts.tag_focus_mode}")
    lines.append(
        "Include meta/artist/copyright: "
        f"{'on' if opts.include_meta else 'off'} / "
        f"{'on' if opts.include_artist else 'off'} / "
        f"{'on' if opts.include_copyright else 'off'}"
    )
    lines.append(
        f"Threshold floor: {opts.min_threshold_floor} (policy)"
    )
    lines.append(f"Trigger tag: {'on' if opts.trigger_tag else 'off'}")
    lines.append(f"Color sanity: {'on' if opts.enable_color_sanity else 'off'} (policy)")

    exclude_tags = set(_parse_tag_list(opts.exclude_tags))
    exclude_regex_raw = _parse_regex_list(opts.exclude_regex)
    if opts.use_normalizer_remove_as_exclude:
        preset_root = opts.normalizer_preset_root
        remove_tags, remove_regex, warn = _load_normalizer_excludes(
            preset_root, opts.normalizer_preset_type, opts.normalizer_preset_file
        )
        if warn:
            lines.append(f"[WARN] {warn}")
        exclude_tags.update(remove_tags)
        exclude_regex_raw.extend(remove_regex)
    exclude_regex = _compile_regex(exclude_regex_raw)
    non_character_regex = _compile_regex(opts.non_character_regex)
    lines.append(f"Exclude tags: {len(exclude_tags)} | Exclude regex: {len(exclude_regex)}")
    lines.append(f"Non-character regex: {len(non_character_regex)}")

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
                    if txt_path.stat().st_size > 0 and txt_path.read_text(encoding="utf-8").strip():
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
    sample_count = 0
    total_images = len(paths)
    progress_step = max(10, min(200, total_images // 20 if total_images > 0 else 10))
    last_progress_ts = 0.0
    lines.append(f"Total images: {total_images}")
    tag_stats: Dict[str, int] = {}
    color_drop_total = 0
    color_drop_images = 0

    def _report_progress(force: bool = False):
        nonlocal last_progress_ts
        if total_images <= 0 or processed <= 0:
            return
        now_ts = time.time()
        if (
            force
            or processed == total_images
            or (processed % progress_step == 0)
            or (now_ts - last_progress_ts) >= 1.5
        ):
            print(f"{processed} out of {total_images} images tagged.")
            last_progress_ts = now_ts

    try:
        from PIL import Image
    except Exception as exc:
        lines.append(f"Failed to load Pillow: {exc}")
        return False, lines

    for batch in _chunked(paths, opts.batch_size):
        images = []
        batch_paths = []
        color_presence_list: List[Optional[Dict[str, float]]] = []
        for path in batch:
            try:
                with Image.open(path) as im:
                    im_rgb = im.convert("RGB")
                    if opts.enable_color_sanity:
                        try:
                            presence = _estimate_color_presence(im_rgb, opts)
                        except Exception:
                            presence = None
                    else:
                        presence = None
                    if wd_fix:
                        im_rgb = _swap_rgb_bgr(im_rgb)
                    images.append(im_rgb)
                    color_presence_list.append(presence)
                batch_paths.append(path)
            except Exception as exc:
                errors += 1
                lines.append(f"[ERROR] {path.name}: {exc}")

        if not images:
            continue

        try:
            if backend == "onnx":
                import numpy as np

                inputs = processor(images=images, return_tensors="np")
                pixel_values = inputs.get("pixel_values")
                if onnx_session is None or onnx_input is None:
                    raise RuntimeError("ONNX session not initialized.")
                outputs = onnx_session.run(None, {onnx_input: pixel_values})
                logits = outputs[0]
                probs = 1.0 / (1.0 + np.exp(-logits))
            else:
                inputs = processor(images=images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    if opts.use_amp and str(device).startswith("cuda"):
                        with torch.cuda.amp.autocast():
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)
                    probs = torch.sigmoid(outputs.logits).cpu().numpy()
        except Exception as exc:
            errors += len(batch_paths)
            lines.append(f"[ERROR] batch failed: {exc}")
            continue

        for idx, (path, row) in enumerate(zip(batch_paths, probs)):
            color_presence = color_presence_list[idx] if idx < len(color_presence_list) else None
            color_debug = [] if opts.debug_color_sanity else None
            tags = _build_tags(
                row,
                labels,
                categories,
                opts,
                category_ids,
                exclude_tags,
                exclude_regex,
                non_character_regex=non_character_regex,
                stats=tag_stats,
                color_presence=color_presence,
                color_debug=color_debug,
            )
            if opts.debug_color_sanity and color_debug:
                color_drop_total += len(color_debug)
                color_drop_images += 1
            tags = _apply_trigger_tag(tags, opts.trigger_tag)
            if opts.skip_empty and not tags:
                processed += 1
                _report_progress()
                continue

            if opts.preview_limit > 0 and sample_count < opts.preview_limit:
                samples.append(f"{path.name}: {', '.join(tags)}")
                if opts.debug_color_sanity and color_debug:
                    for item in color_debug:
                        samples.append(f"  [color_sanity] dropped {item}")
                sample_count += 1

            if opts.preview_only:
                processed += 1
                _report_progress()
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
                if opts.keep_existing_tags:
                    merged_main = list(existing_main) + tags
                else:
                    merged_main = tags

            if opts.dedupe:
                merged_main = _dedup_preserve(merged_main)
            if opts.sort_tags:
                merged_main = sorted(merged_main)
            merged_main = _apply_trigger_tag(merged_main, opts.trigger_tag)

            if merged_main or not opts.skip_empty:
                text = _format_tag_file(
                    merged_main,
                    existing_optional,
                    existing_warning,
                    opts.newline_end,
                    opts.strip_whitespace,
                )
                txt_path.write_text(text, encoding="utf-8")
                written += 1

            processed += 1
            _report_progress()

    _report_progress(force=True)

    if samples:
        lines.append("Sample tags:")
        lines.extend(samples)

    if tag_stats:
        lines.append(
            "Tag stats: "
            f"general={tag_stats.get('general', 0)}, "
            f"character={tag_stats.get('character', 0)}, "
            f"meta={tag_stats.get('meta', 0)}, "
            f"subject_general={tag_stats.get('subject_general', 0)}, "
            f"non_character_general={tag_stats.get('non_character_general', 0)}, "
            f"rating={tag_stats.get('rating', 0)}, "
            f"images_with_general={tag_stats.get('images_with_general', 0)}, "
            f"images_with_character={tag_stats.get('images_with_character', 0)}"
        )
    if opts.debug_color_sanity:
        lines.append(
            f"Color sanity drops: {color_drop_total} tags across {color_drop_images} images."
        )

    if opts.preview_only:
        lines.append(f"Preview done. Processed {processed} images (no files written).")
    else:
        lines.append(
            f"Done. Processed {processed} images, wrote {written} tag files, skipped {skipped}, errors {errors}."
        )

    return True, lines


def handle(form, ctx):
    active_tab = "offline_tagger"

    raw_folder = (form.get("folder") or "").strip()
    if not raw_folder:
        return build_tool_result(
            active_tab,
            ["Dataset folder is required."],
            ok=False,
            error="Dataset folder is required.",
        )

    dataset_path = readable_path(raw_folder)
    if not dataset_path.exists() or not dataset_path.is_dir():
        return build_tool_result(
            active_tab,
            [f"Dataset folder not found: {dataset_path}"],
            ok=False,
            error=f"Dataset folder not found: {dataset_path}",
        )

    form_opts = dict(form) if isinstance(form, dict) else {k: form.get(k) for k in form.keys()}
    form_opts["input_dir"] = raw_folder
    for key in (
        "recursive",
        "include_character",
        "include_rating",
        "preview_only",
        "dedupe",
        "sort_tags",
        "keep_existing_tags",
        "newline_end",
        "strip_whitespace",
    ):
        if key not in form_opts:
            form_opts[key] = None
    deprecated = _find_deprecated_keys(form_opts)
    opts = _effective_opts(form_opts, TAGGER_POLICY)
    ok, lines = run_tagger(opts, deprecated_keys=deprecated)
    return build_tool_result(
        active_tab,
        lines,
        ok=ok,
        error="" if ok else "Offline tagger failed. See logs for details.",
    )


def _cli():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Offline tagger policy summary."
    )
    parser.add_argument("--dump", action="store_true", help="Print policy and exit")
    args = parser.parse_args()

    if args.dump:
        print(TAGGER_POLICY)
        sys.exit(0)
    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    _cli()
