from dataclasses import dataclass, field
import fnmatch
import json
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import time
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from services.offline_tagger_rules import (
    BUCKET_APPEARANCE_IDENTITY,
    BUCKET_BACKGROUND_PLACE,
    BUCKET_CAMERA_COMPOSITION,
    BUCKET_CLOTHING_OUTFIT,
    BUCKET_LIMB_ACTION,
    BUCKET_LIGHTING_ENVIRONMENT,
    BUCKET_OBJECT_PROP,
    BUCKET_POSE_ACTION,
    BUCKET_UNKNOWN,
    COMPILED_REGEX_ALLOW_BACKGROUND,
    COMPILED_REGEX_ALLOW_CAMERA,
    COMPILED_REGEX_ALLOW_LIGHTING,
    COMPILED_REGEX_ALLOW_LIMB,
    COMPILED_REGEX_ALLOW_OBJECT,
    COMPILED_REGEX_ALLOW_POSE,
    COMPILED_REGEX_DENY_APPEARANCE,
    COMPILED_REGEX_DENY_CLOTHING,
    EXACT_ALLOW_BACKGROUND,
    EXACT_ALLOW_CAMERA,
    EXACT_ALLOW_LIGHTING,
    EXACT_ALLOW_LIMB,
    EXACT_ALLOW_OBJECT,
    EXACT_ALLOW_POSE,
    EXACT_DENY_APPEARANCE,
    EXACT_DENY_CLOTHING,
    normalize_rule_tag,
)
from services import tag_policy
from utils.io import readable_path
from utils.dataset import split_tags, join_tags
from utils.tags import normalize_caption_tags, tag_compare_key, to_caption_tag
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
DEFAULT_OUTPUT_PROFILE = "background_pose_only"
DEFAULT_SIMPLE_OUTPUT_PROFILE = "guided_flow_strict"
DEFAULT_TAG_STRENGTH = 50
DEFAULT_MAX_AUTO_TAGS = 24

DEFAULT_MAX_GENERAL_TAGS = 0
DEFAULT_MAX_CHARACTER_TAGS = 0
DEFAULT_MAX_META_TAGS = 0

DEFAULT_MCUT_RELAX_GENERAL = 0.08
DEFAULT_MCUT_RELAX_CHARACTER = 0.02
DEFAULT_MCUT_RELAX_META = 0.05
DEFAULT_MCUT_MIN_GENERAL_TAGS = 8
DEFAULT_POLICY_MCUT_MIN_GENERAL_TAGS = 0
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
DEFAULT_ENABLE_DANBOORU_SAFENET = False
DEFAULT_DANBOORU_SAFENET_MAX_LOOKUPS = 120
DEFAULT_SELECTIVE_MIN_KEEP_TAGS = 2
DEFAULT_SELECTIVE_UNKNOWN_FALLBACK_MAX = 2

DEFAULT_NON_CHARACTER_REGEX = [
    r"(?:^|[ _])(background|scenery|landscape|cityscape)(?:$|[ _])",
    r"(?:^|[ _])(indoors|outdoors|sky|cloud|sunset|sunrise|moon|star|night|day)(?:$|[ _])",
    r"(?:^|[ _])(room|bedroom|bathroom|kitchen|classroom|office|library|corridor|hallway)(?:$|[ _])",
    r"(?:^|[ _])(street|road|alley|bridge|sidewalk|building|window|door)(?:$|[ _])",
    r"(?:^|[ _])(forest|tree|grass|flower|mountain|river|lake|sea|ocean|beach)(?:$|[ _])",
    r"(?:^|[ _])(car|bus|train|airplane|ship|bicycle)(?:$|[ _])",
    r" background$",
]

OUTPUT_PROFILE_STANDARD_FULL = "standard_full"
OUTPUT_PROFILE_BACKGROUND_POSE_ONLY = "background_pose_only"
OUTPUT_PROFILE_CUSTOM_SELECTIVE = "custom_selective"
OUTPUT_PROFILE_GUIDED_FLOW_STRICT = "guided_flow_strict"
VALID_OUTPUT_PROFILES = {
    OUTPUT_PROFILE_STANDARD_FULL,
    OUTPUT_PROFILE_BACKGROUND_POSE_ONLY,
    OUTPUT_PROFILE_CUSTOM_SELECTIVE,
    OUTPUT_PROFILE_GUIDED_FLOW_STRICT,
}

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
    "ui_mode": "legacy",
    "tag_strength": DEFAULT_TAG_STRENGTH,
    "min_threshold_floor": DEFAULT_MIN_THRESHOLD_FLOOR,
    "mcut_relax_general": DEFAULT_MCUT_RELAX_GENERAL,
    "mcut_relax_character": DEFAULT_MCUT_RELAX_CHARACTER,
    "mcut_relax_meta": DEFAULT_MCUT_RELAX_META,
    "mcut_min_general_tags": DEFAULT_MCUT_MIN_GENERAL_TAGS,
    "policy_mcut_min_general_tags": DEFAULT_POLICY_MCUT_MIN_GENERAL_TAGS,
    "mcut_min_character_tags": DEFAULT_MCUT_MIN_CHARACTER_TAGS,
    "mcut_min_meta_tags": DEFAULT_MCUT_MIN_META_TAGS,
    "output_profile": DEFAULT_OUTPUT_PROFILE,
    "max_auto_tags": DEFAULT_MAX_AUTO_TAGS,
    "selective_keep_background_place": True,
    "selective_keep_object_prop": True,
    "selective_keep_pose_action": True,
    "selective_keep_appearance": False,
    "selective_keep_clothing": False,
    "selective_keep_character_names": False,
    "selective_keep_artist_copyright": False,
    "selective_keep_rating_meta": False,
    "selective_keep_unknown_general": False,
    "tag_focus_mode": DEFAULT_TAG_FOCUS_MODE,
    "include_general": True,
    "include_character": True,
    "include_rating": False,
    "include_meta": False,
    "include_copyright": False,
    "include_artist": False,
    "replace_underscore": False,
    "tag_policy": tag_policy.DEFAULT_TAG_POLICY,
    "policy_keep_tags": "",
    "policy_block_tags": "",
    "policy_block_regex": "",
    "block_permanent_marks": False,
    "replace_existing_captions": True,
    "write_mode": "overwrite",
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
    "prefix_tags": "",
    "blocked_tags": "",
    "backup_existing": True,
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
    "danbooru_safenet": DEFAULT_ENABLE_DANBOORU_SAFENET,
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

_MODEL_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_DOWNLOAD_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.csv"]
_WORKER_ENV_FLAG = "BATCHBENCH_OFFLINE_TAGGER_WORKER"


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
    simple_mode: bool
    tag_strength: int
    min_threshold_floor: float
    mcut_relax_general: float
    mcut_relax_character: float
    mcut_relax_meta: float
    mcut_min_general_tags: int
    policy_mcut_min_general_tags: int
    mcut_min_character_tags: int
    mcut_min_meta_tags: int
    output_profile: str
    max_auto_tags: int
    selective_keep_background_place: bool
    selective_keep_object_prop: bool
    selective_keep_pose_action: bool
    selective_keep_appearance: bool
    selective_keep_clothing: bool
    selective_keep_character_names: bool
    selective_keep_artist_copyright: bool
    selective_keep_rating_meta: bool
    selective_keep_unknown_general: bool
    tag_focus_mode: str
    include_general: bool
    include_character: bool
    include_rating: bool
    include_meta: bool
    include_copyright: bool
    include_artist: bool
    replace_underscore: bool
    tag_policy: str
    policy_keep_tags: List[str]
    policy_block_tags: List[str]
    policy_block_regex: List[str]
    block_permanent_marks: bool
    replace_existing_captions: bool
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
    blocked_tags: List[str]
    exclude_regex: List[str]
    non_character_regex: List[str]
    use_normalizer_remove_as_exclude: bool
    backend: str
    use_amp: bool
    trigger_tag: str
    prefix_tags: List[str]
    backup_existing: bool
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
    danbooru_safenet: bool


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


def tag_strength_to_threshold(strength: Any) -> float:
    value = _parse_int(strength, DEFAULT_TAG_STRENGTH)
    value = max(0, min(value, 100))
    threshold = 0.50 - (value / 100.0 * 0.20)
    return round(max(0.30, min(threshold, 0.50)), 6)


def _normalize_user_tag(tag: str) -> str:
    return normalize_rule_tag(tag)


def _parse_normalized_tag_list(raw: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for tag in _parse_tag_list(raw):
        norm = _normalize_user_tag(tag)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


_BLOCKED_PATTERN_NORM_RE = re.compile(r"[^a-z0-9_*]+")


def _normalize_blocked_pattern(pattern: str) -> str:
    text = (pattern or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = _BLOCKED_PATTERN_NORM_RE.sub("_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _parse_blocked_tag_patterns(raw: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for tag in _parse_tag_list(raw):
        norm = _normalize_blocked_pattern(tag)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


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


def _tagger_options_to_payload(opts: TaggerOptions) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in vars(opts).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, list):
            payload[key] = list(value)
        else:
            payload[key] = value
    return payload


def _tagger_options_from_payload(payload: Dict[str, Any]) -> TaggerOptions:
    data = dict(payload or {})
    data["dataset_path"] = readable_path(str(data.get("dataset_path") or "."))
    root = data.get("normalizer_preset_root")
    data["normalizer_preset_root"] = readable_path(str(root)) if root else None
    return TaggerOptions(**data)


def _candidate_source_roots() -> List[Path]:
    roots: List[Path] = []
    for raw in (
        os.environ.get("BATCHBENCH_SOURCE_ROOT"),
        os.environ.get("BATCHBENCH_DATA_DIR"),
    ):
        if raw:
            roots.append(readable_path(raw))
    exe_dir = Path(sys.executable).resolve().parent
    roots.extend(
        [
            Path.cwd(),
            exe_dir,
            exe_dir.parent,
            exe_dir.parent.parent,
        ]
    )
    out: List[Path] = []
    seen = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            resolved = root.absolute()
        key = str(resolved).lower() if os.name == "nt" else str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if (resolved / "services" / "offline_tagger.py").exists():
            out.append(resolved)
    return out


def _external_worker_python(source_root: Path) -> Optional[Path]:
    override = os.environ.get("BATCHBENCH_OFFLINE_TAGGER_PYTHON")
    candidates = [readable_path(override)] if override else []
    candidates.extend(
        [
            source_root / ".venv" / "Scripts" / "python.exe",
            source_root / "venv" / "Scripts" / "python.exe",
        ]
    )
    current = Path(sys.executable).resolve()
    for candidate in candidates:
        if candidate and candidate.exists() and candidate.resolve() != current:
            return candidate
    return None


def _run_external_worker(
    opts: TaggerOptions,
    deprecated_keys: Optional[List[str]],
) -> Optional[Tuple[bool, List[str]]]:
    if not getattr(sys, "frozen", False):
        return None
    if os.environ.get(_WORKER_ENV_FLAG) == "1":
        return None
    for source_root in _candidate_source_roots():
        python_exe = _external_worker_python(source_root)
        if not python_exe:
            continue
        worker_root = Path(tempfile.gettempdir())
        payload_path = worker_root / f"offline_tagger_worker_{os.getpid()}_{int(time.time())}.json"
        result_path = worker_root / f"offline_tagger_worker_result_{os.getpid()}_{int(time.time())}.json"
        payload = {
            "options": _tagger_options_to_payload(opts),
            "deprecated_keys": list(deprecated_keys or []),
        }
        try:
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
            env = dict(os.environ)
            env[_WORKER_ENV_FLAG] = "1"
            proc = subprocess.run(
                [
                    str(python_exe),
                    "-m",
                    "services.offline_tagger",
                    "--run-options",
                    str(payload_path),
                    "--result",
                    str(result_path),
                ],
                cwd=str(source_root),
                env=env,
                text=True,
                capture_output=True,
            )
            if result_path.exists():
                data = json.loads(result_path.read_text(encoding="utf-8"))
                lines = list(data.get("lines") or [])
                lines.insert(0, f"Offline Tagger worker: {python_exe}")
                lines.insert(1, f"Worker source root: {source_root}")
                if proc.stdout.strip():
                    lines.append("Worker stdout:")
                    lines.extend(proc.stdout.strip().splitlines()[-20:])
                if proc.stderr.strip():
                    lines.append("Worker stderr:")
                    lines.extend(proc.stderr.strip().splitlines()[-20:])
                return bool(data.get("ok")), lines
            return (
                False,
                [
                    f"Offline Tagger worker failed before writing result: {python_exe}",
                    f"Exit code: {proc.returncode}",
                    f"stdout: {proc.stdout.strip()}",
                    f"stderr: {proc.stderr.strip()}",
                ],
            )
        except Exception as exc:
            return (
                False,
                [
                    f"Offline Tagger worker launch failed: {python_exe}",
                    f"{type(exc).__name__}: {exc}",
                ],
            )
        finally:
            for path in (payload_path, result_path):
                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass
    return None


def _effective_opts(form_opts: Dict[str, Any], policy: Dict[str, Any]) -> TaggerOptions:
    policy = policy or TAGGER_POLICY
    raw_input = (form_opts.get("input_dir") or form_opts.get("folder") or "").strip()
    dataset_path = readable_path(raw_input) if raw_input else Path(".")
    replace_present = "replace_existing_captions" in form_opts and form_opts.get("replace_existing_captions") not in (None, "")
    if replace_present:
        replace_existing_captions = _parse_bool(form_opts.get("replace_existing_captions"))
        write_mode = "overwrite" if replace_existing_captions else "skip"
    else:
        write_mode = (form_opts.get("write_mode") or policy.get("write_mode") or "overwrite").strip().lower()
        replace_existing_captions = write_mode in {"overwrite", "replace"}
    if write_mode == "skip_if_exists":
        write_mode = "skip"
    if write_mode not in {"overwrite", "append", "skip"}:
        write_mode = "overwrite"
    replace_existing_captions = write_mode == "overwrite"

    def _fallback(key: str, default_key: Optional[str] = None):
        if key in form_opts and form_opts.get(key) not in (None, ""):
            return form_opts.get(key)
        if default_key and default_key in form_opts and form_opts.get(default_key) not in (None, ""):
            return form_opts.get(default_key)
        return policy.get(key)

    ui_mode = str(_fallback("ui_mode") or "").strip().lower()
    simple_mode = ui_mode in {"simple", "auto_tag_assist", "guided_flow_strict"} or _parse_bool(
        _fallback("simple_mode")
    )
    tag_strength = max(0, min(_parse_int(_fallback("tag_strength"), DEFAULT_TAG_STRENGTH), 100))

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
    selected_tag_policy = (
        form_opts.get("tag_policy")
        or form_opts.get("policy_name")
        or policy.get("tag_policy")
        or tag_policy.DEFAULT_TAG_POLICY
    ).strip().lower()
    if selected_tag_policy not in tag_policy.PROFILES:
        selected_tag_policy = tag_policy.DEFAULT_TAG_POLICY
    output_profile = (
        form_opts.get("output_profile")
        or policy.get("output_profile")
        or DEFAULT_OUTPUT_PROFILE
    ).strip().lower()
    if output_profile not in VALID_OUTPUT_PROFILES:
        output_profile = DEFAULT_OUTPUT_PROFILE
    max_auto_tags = max(
        1,
        min(_parse_int(_fallback("max_auto_tags"), int(policy.get("max_auto_tags", DEFAULT_MAX_AUTO_TAGS))), 100),
    )
    cleanup_keys = {
        "remove_appearance_identity",
        "remove_outfit_accessory",
        "remove_character_names",
        "remove_artist_copyright",
        "remove_rating_meta",
        "remove_unclassified",
    }
    explicit_simple_cleanup = bool(
        simple_mode and any(key in form_opts and form_opts.get(key) is not None for key in cleanup_keys)
    )
    if simple_mode:
        general_threshold = tag_strength_to_threshold(tag_strength)
        threshold_mode = "fixed"
        output_profile = OUTPUT_PROFILE_GUIDED_FLOW_STRICT
    selective_keep_background_place = (
        _parse_bool(_fallback("selective_keep_background_place"))
        if "selective_keep_background_place" in form_opts
        else bool(policy.get("selective_keep_background_place", True))
    )
    selective_keep_object_prop = (
        _parse_bool(_fallback("selective_keep_object_prop"))
        if "selective_keep_object_prop" in form_opts
        else bool(policy.get("selective_keep_object_prop", True))
    )
    selective_keep_pose_action = (
        _parse_bool(_fallback("selective_keep_pose_action"))
        if "selective_keep_pose_action" in form_opts
        else bool(policy.get("selective_keep_pose_action", True))
    )
    selective_keep_appearance = (
        _parse_bool(_fallback("selective_keep_appearance"))
        if "selective_keep_appearance" in form_opts
        else bool(policy.get("selective_keep_appearance", False))
    )
    selective_keep_clothing = (
        _parse_bool(_fallback("selective_keep_clothing"))
        if "selective_keep_clothing" in form_opts
        else bool(policy.get("selective_keep_clothing", False))
    )
    selective_keep_character_names = (
        _parse_bool(_fallback("selective_keep_character_names"))
        if "selective_keep_character_names" in form_opts
        else bool(policy.get("selective_keep_character_names", False))
    )
    selective_keep_artist_copyright = (
        _parse_bool(_fallback("selective_keep_artist_copyright"))
        if "selective_keep_artist_copyright" in form_opts
        else bool(policy.get("selective_keep_artist_copyright", False))
    )
    selective_keep_rating_meta = (
        _parse_bool(_fallback("selective_keep_rating_meta"))
        if "selective_keep_rating_meta" in form_opts
        else bool(policy.get("selective_keep_rating_meta", False))
    )
    selective_keep_unknown_general = (
        _parse_bool(_fallback("selective_keep_unknown_general"))
        if "selective_keep_unknown_general" in form_opts
        else bool(policy.get("selective_keep_unknown_general", False))
    )
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
    include_meta = _parse_bool(_fallback("include_meta")) if "include_meta" in form_opts else bool(
        policy.get("include_meta", False)
    )
    include_copyright = _parse_bool(_fallback("include_copyright")) if "include_copyright" in form_opts else bool(
        policy.get("include_copyright", False)
    )
    include_artist = _parse_bool(_fallback("include_artist")) if "include_artist" in form_opts else bool(
        policy.get("include_artist", False)
    )
    if simple_mode:
        include_general = True
        include_character = False
        include_rating = False
        include_meta = False
        include_copyright = False
        include_artist = False
    if explicit_simple_cleanup:
        if form_opts.get("min_general") not in (None, "") or form_opts.get("general_threshold") not in (None, ""):
            general_threshold = max(
                0.01,
                min(
                    _parse_float(
                        _fallback("min_general", "general_threshold"),
                        tag_strength_to_threshold(tag_strength),
                    ),
                    0.99,
                ),
            )
        remove_appearance = _parse_bool(_fallback("remove_appearance_identity"))
        remove_clothing = _parse_bool(_fallback("remove_outfit_accessory"))
        remove_character = _parse_bool(_fallback("remove_character_names"))
        remove_artist_copyright = _parse_bool(_fallback("remove_artist_copyright"))
        remove_rating_meta = _parse_bool(_fallback("remove_rating_meta"))
        remove_unknown = _parse_bool(_fallback("remove_unclassified"))
        selective_keep_background_place = True
        selective_keep_object_prop = True
        selective_keep_pose_action = True
        selective_keep_appearance = not remove_appearance
        selective_keep_clothing = not remove_clothing
        selective_keep_character_names = not remove_character
        selective_keep_artist_copyright = not remove_artist_copyright
        selective_keep_rating_meta = not remove_rating_meta
        selective_keep_unknown_general = not remove_unknown
        include_character = not remove_character
        include_rating = not remove_rating_meta
        include_meta = not remove_rating_meta
        include_copyright = not remove_artist_copyright
        include_artist = not remove_artist_copyright
        has_general_cleanup = remove_appearance or remove_clothing or remove_unknown
        output_profile = OUTPUT_PROFILE_CUSTOM_SELECTIVE if has_general_cleanup else OUTPUT_PROFILE_STANDARD_FULL

    dedupe = _parse_bool(_fallback("dedupe")) if "dedupe" in form_opts else bool(policy.get("dedupe", True))
    sort_tags = _parse_bool(_fallback("sort_tags")) if "sort_tags" in form_opts else bool(policy.get("sort_tags", True))
    keep_existing_tags = (
        _parse_bool(_fallback("keep_existing_tags"))
        if "keep_existing_tags" in form_opts
        else bool(policy.get("keep_existing_tags", True))
    )
    if write_mode != "append":
        keep_existing_tags = False
    if simple_mode:
        dedupe = True
        sort_tags = False
        keep_existing_tags = write_mode == "append"

    prefix_tags = _parse_normalized_tag_list(_fallback("prefix_tags") or "")
    legacy_trigger = (form_opts.get("trigger_tag") or policy.get("trigger_tag") or "").strip()
    if legacy_trigger:
        prefix_tags = _dedup_preserve(_parse_normalized_tag_list(legacy_trigger) + prefix_tags)
    blocked_tags = _parse_blocked_tag_patterns(_fallback("blocked_tags") or _fallback("exclude_tags") or "")
    backup_existing = parse_bool(
        _fallback("backup_existing"),
        default=bool(policy.get("backup_existing", True)),
    )
    if write_mode == "overwrite":
        backup_existing = True
    policy_keep_tags = _parse_normalized_tag_list(_fallback("policy_keep_tags") or _fallback("custom_policy_keep") or "")
    policy_block_tags = _parse_normalized_tag_list(_fallback("policy_block_tags") or _fallback("custom_policy_block") or "")
    policy_block_regex = _parse_regex_list(
        _fallback("policy_block_regex") or _fallback("custom_policy_block_regex") or ""
    )
    block_permanent_marks = (
        _parse_bool(_fallback("block_permanent_marks"))
        if "block_permanent_marks" in form_opts
        else bool(policy.get("block_permanent_marks", False))
    )
    policy_min_default = (
        DEFAULT_POLICY_MCUT_MIN_GENERAL_TAGS
        if selected_tag_policy == tag_policy.POLICY_CHARACTER_IDENTITY_OMITTED
        else int(policy.get("mcut_min_general_tags", DEFAULT_MCUT_MIN_GENERAL_TAGS))
    )

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
        simple_mode=simple_mode,
        tag_strength=tag_strength,
        min_threshold_floor=_parse_float(
            _fallback("min_threshold_floor"),
            float(policy.get("min_threshold_floor", DEFAULT_MIN_THRESHOLD_FLOOR)),
        ),
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
        policy_mcut_min_general_tags=max(
            0,
            _parse_int(_fallback("policy_mcut_min_general_tags"), policy_min_default),
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
        output_profile=output_profile,
        max_auto_tags=max_auto_tags,
        selective_keep_background_place=selective_keep_background_place,
        selective_keep_object_prop=selective_keep_object_prop,
        selective_keep_pose_action=selective_keep_pose_action,
        selective_keep_appearance=selective_keep_appearance,
        selective_keep_clothing=selective_keep_clothing,
        selective_keep_character_names=selective_keep_character_names,
        selective_keep_artist_copyright=selective_keep_artist_copyright,
        selective_keep_rating_meta=selective_keep_rating_meta,
        selective_keep_unknown_general=selective_keep_unknown_general,
        tag_focus_mode=tag_focus_mode,
        include_general=include_general,
        include_character=include_character,
        include_rating=include_rating,
        include_meta=include_meta,
        include_copyright=include_copyright,
        include_artist=include_artist,
        replace_underscore=_parse_bool(_fallback("replace_underscore"))
        if "replace_underscore" in form_opts
        else bool(policy.get("replace_underscore", False)),
        tag_policy=selected_tag_policy,
        policy_keep_tags=policy_keep_tags,
        policy_block_tags=policy_block_tags,
        policy_block_regex=policy_block_regex,
        block_permanent_marks=block_permanent_marks,
        replace_existing_captions=replace_existing_captions,
        write_mode=write_mode,
        preview_only=_parse_bool(_fallback("preview_only"))
        if "preview_only" in form_opts
        else bool(policy.get("preview_only", False)),
        preview_limit=preview_limit,
        limit=limit,
        max_tags=max(0, _parse_int(_fallback("max_tags"), int(policy.get("max_tags", 0)))),
        max_general_tags=max_auto_tags
        if simple_mode
        else max(0, _parse_int(_fallback("max_general_tags"), int(policy.get("max_general_tags", 0)))),
        max_character_tags=max(
            0, _parse_int(_fallback("max_character_tags"), int(policy.get("max_character_tags", 0)))
        ),
        max_meta_tags=max(0, _parse_int(_fallback("max_meta_tags"), int(policy.get("max_meta_tags", 0)))),
        character_topk=max(0, _parse_int(_fallback("character_topk"), int(policy.get("character_topk", 0)))),
        skip_empty=_parse_bool(_fallback("skip_empty")) if "skip_empty" in form_opts else bool(policy.get("skip_empty", True)),
        local_only=_parse_bool(_fallback("local_only")) if "local_only" in form_opts else bool(policy.get("local_only", False)),
        exclude_tags=[]
        if simple_mode
        else _parse_tag_list(_fallback("exclude_tags") or policy.get("exclude_tags") or ""),
        blocked_tags=blocked_tags,
        exclude_regex=_parse_regex_list(_fallback("exclude_regex") or policy.get("exclude_regex") or []),
        non_character_regex=_parse_regex_list(
            _fallback("non_character_regex") or policy.get("non_character_regex") or DEFAULT_NON_CHARACTER_REGEX
        ),
        use_normalizer_remove_as_exclude=bool(policy.get("use_normalizer_remove_as_exclude", False)),
        backend=(_fallback("backend") or DEFAULT_BACKEND).strip().lower(),
        use_amp=_parse_bool(_fallback("use_amp")) if "use_amp" in form_opts else bool(policy.get("use_amp", False)),
        trigger_tag=legacy_trigger,
        prefix_tags=prefix_tags,
        backup_existing=backup_existing,
        dedupe=dedupe,
        sort_tags=sort_tags,
        keep_existing_tags=keep_existing_tags,
        newline_end=_parse_bool(_fallback("newline_end")) if "newline_end" in form_opts else bool(policy.get("newline_end", True)),
        strip_whitespace=_parse_bool(_fallback("strip_whitespace")) if "strip_whitespace" in form_opts else bool(policy.get("strip_whitespace", True)),
        force_wd_bgr_fix=bool(policy.get("force_wd_bgr_fix", True)),
        general_category_id=None,
        character_category_id=None,
        rating_category_id=None,
        normalizer_preset_root=readable_path(str(policy.get("normalizer_preset_root")))
        if policy.get("normalizer_preset_root")
        else None,
        normalizer_preset_type=str(policy.get("normalizer_preset_type") or ""),
        normalizer_preset_file=str(policy.get("normalizer_preset_file") or ""),
        enable_color_sanity=_parse_bool(_fallback("enable_color_sanity"))
        if "enable_color_sanity" in form_opts
        else bool(policy.get("enable_color_sanity", DEFAULT_ENABLE_COLOR_SANITY)),
        color_ratio_threshold=_parse_float(_fallback("color_ratio_threshold"), DEFAULT_COLOR_RATIO_THRESHOLD),
        color_min_saturation=_parse_float(_fallback("color_min_saturation"), DEFAULT_COLOR_MIN_SATURATION),
        color_min_value=_parse_float(_fallback("color_min_value"), DEFAULT_COLOR_MIN_VALUE),
        color_keep_if_score_ge=_parse_float(_fallback("color_keep_if_score_ge"), DEFAULT_COLOR_KEEP_IF_SCORE_GE),
        color_downscale=max(16, _parse_int(_fallback("color_downscale"), DEFAULT_COLOR_DOWNSCALE)),
        debug_color_sanity=_parse_bool(_fallback("debug_color_sanity"))
        if "debug_color_sanity" in form_opts
        else bool(policy.get("debug_color_sanity", DEFAULT_DEBUG_COLOR_SANITY)),
        danbooru_safenet=_parse_bool(_fallback("danbooru_safenet"))
        if "danbooru_safenet" in form_opts
        else bool(policy.get("danbooru_safenet", DEFAULT_ENABLE_DANBOORU_SAFENET)),
    )


def _iter_images(root: Path, recursive: bool, exts: List[str]) -> List[Path]:
    exts_set = {e.lower() for e in exts}
    iterator = root.rglob("*") if recursive else root.iterdir()
    out = []
    for p in iterator:
        if not p.is_file():
            continue
        try:
            rel_parts = {part.lower() for part in p.relative_to(root).parts}
            if ".batchbench_backup" in rel_parts:
                continue
        except Exception:
            pass
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


def _cache_lookup_model_bundle(model_id: str, device: str, backend: str) -> Optional[Dict[str, Any]]:
    key = (model_id, device, backend)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    requested = (device or "").strip().lower()
    if requested in {"", "auto"}:
        for candidate_device in ("cuda", "cpu", "mps"):
            cached = _MODEL_CACHE.get((model_id, candidate_device, backend))
            if cached is not None:
                return cached
    return None


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

    local_dir = None
    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            allow_patterns=_DOWNLOAD_PATTERNS,
            local_files_only=True,
        )
    except Exception:
        local_dir = None

    if local_dir is None:
        if local_only:
            raise RuntimeError("Model files not found locally. Disable local-only to download.")
        try:
            local_dir = snapshot_download(
                repo_id=model_id,
                allow_patterns=_DOWNLOAD_PATTERNS,
                local_files_only=False,
            )
        except Exception as exc:
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


def _torch_import_diagnostics() -> str:
    parts: List[str] = []
    exe_dir = Path(sys.executable).resolve().parent
    root = Path(getattr(sys, "_MEIPASS", exe_dir))
    parts.append(f"executable={sys.executable}")
    parts.append(f"_MEIPASS={getattr(sys, '_MEIPASS', '')}")
    candidates = [
        root / "torch" / "lib",
        root / "_internal" / "torch" / "lib",
        exe_dir / "_internal" / "torch" / "lib",
    ]
    dll_names = ["c10.dll", "torch_cpu.dll", "torch_python.dll", "torch_global_deps.dll", "libiomp5md.dll"]
    for candidate in candidates:
        parts.append(f"torch_lib={candidate} exists={candidate.exists()}")
        if not candidate.exists():
            continue
        for dll_name in dll_names:
            dll_path = candidate / dll_name
            if not dll_path.exists():
                parts.append(f"{dll_name}=missing")
                continue
            try:
                import ctypes

                ctypes.WinDLL(str(dll_path))
                parts.append(f"{dll_name}=load_ok")
            except Exception as exc:
                parts.append(f"{dll_name}=load_failed:{type(exc).__name__}:{exc}")
    return " | ".join(parts)


def _load_model_bundle(model_id: str, device: str, local_only: bool, backend: str):
    backend = (backend or DEFAULT_BACKEND).strip().lower()
    cache_key = (model_id, device, backend)
    cached_bundle = _cache_lookup_model_bundle(model_id, device, backend)
    if cached_bundle is not None:
        return cached_bundle

    try:
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Offline Tagger dependencies "
            f"(transformers/torch stack): {type(exc).__name__}: {exc}. "
            "If this is the EXE build, rebuild it with compile_exe.bat so PyInstaller bundles the ML packages. "
            f"Diagnostics: {_torch_import_diagnostics()}"
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
                "Failed to import torch for the Offline Tagger: "
                f"{type(exc).__name__}: {exc}. "
                "If this is the EXE build, rebuild it with compile_exe.bat. "
                f"Diagnostics: {_torch_import_diagnostics()}"
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
        cached = _MODEL_CACHE[actual_key]
        _MODEL_CACHE[cache_key] = cached
        return cached

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
    _MODEL_CACHE[cache_key] = bundle
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


@dataclass
class SelectiveRules:
    output_profile: str
    general_allow_buckets: set
    include_character: bool
    include_rating: bool
    include_meta: bool
    include_artist: bool
    include_copyright: bool
    drop_unknown_general: bool
    use_unknown_fallback: bool = True

    @property
    def include_rating_meta(self) -> bool:
        return bool(self.include_rating or self.include_meta or self.include_artist or self.include_copyright)


@dataclass
class DanbooruSafeNetState:
    enabled: bool
    max_lookups: int
    lookups: int = 0
    resolved: int = 0
    errors: int = 0
    skipped: int = 0
    cache: Dict[str, str] = field(default_factory=dict)


def _resolve_output_profile(raw: str) -> str:
    profile = (raw or "").strip().lower()
    if profile in VALID_OUTPUT_PROFILES:
        return profile
    return DEFAULT_OUTPUT_PROFILE


def _resolve_selective_rules(opts: TaggerOptions) -> Optional[SelectiveRules]:
    profile = _resolve_output_profile(opts.output_profile)
    if profile == OUTPUT_PROFILE_STANDARD_FULL:
        return None
    if profile == OUTPUT_PROFILE_BACKGROUND_POSE_ONLY:
        return SelectiveRules(
            output_profile=profile,
            general_allow_buckets={
                BUCKET_BACKGROUND_PLACE,
                BUCKET_OBJECT_PROP,
                BUCKET_POSE_ACTION,
                BUCKET_LIMB_ACTION,
            },
            include_character=False,
            include_rating=False,
            include_meta=False,
            include_artist=False,
            include_copyright=False,
            drop_unknown_general=True,
        )
    if profile == OUTPUT_PROFILE_GUIDED_FLOW_STRICT:
        return SelectiveRules(
            output_profile=profile,
            general_allow_buckets={
                BUCKET_BACKGROUND_PLACE,
                BUCKET_OBJECT_PROP,
                BUCKET_POSE_ACTION,
                BUCKET_LIMB_ACTION,
                BUCKET_CAMERA_COMPOSITION,
                BUCKET_LIGHTING_ENVIRONMENT,
            },
            include_character=False,
            include_rating=False,
            include_meta=False,
            include_artist=False,
            include_copyright=False,
            drop_unknown_general=True,
            use_unknown_fallback=False,
        )
    allow_buckets = set()
    if opts.selective_keep_background_place:
        allow_buckets.add(BUCKET_BACKGROUND_PLACE)
    if opts.selective_keep_object_prop:
        allow_buckets.add(BUCKET_OBJECT_PROP)
    if opts.selective_keep_pose_action:
        allow_buckets.add(BUCKET_POSE_ACTION)
        allow_buckets.add(BUCKET_LIMB_ACTION)
    if opts.selective_keep_appearance:
        allow_buckets.add(BUCKET_APPEARANCE_IDENTITY)
    if opts.selective_keep_clothing:
        allow_buckets.add(BUCKET_CLOTHING_OUTFIT)
    if getattr(opts, "simple_mode", False):
        allow_buckets.add(BUCKET_CAMERA_COMPOSITION)
        allow_buckets.add(BUCKET_LIGHTING_ENVIRONMENT)
    return SelectiveRules(
        output_profile=profile,
        general_allow_buckets=allow_buckets,
        include_character=bool(opts.selective_keep_character_names),
        include_rating=bool(opts.selective_keep_rating_meta),
        include_meta=bool(opts.selective_keep_rating_meta),
        include_artist=bool(getattr(opts, "selective_keep_artist_copyright", False)),
        include_copyright=bool(getattr(opts, "selective_keep_artist_copyright", False)),
        drop_unknown_general=not bool(opts.selective_keep_unknown_general),
        use_unknown_fallback=not bool(getattr(opts, "simple_mode", False)),
    )


def _enabled_bucket_labels(rules: Optional[SelectiveRules]) -> str:
    if rules is None:
        return "all (standard_full)"
    ordered = [
        BUCKET_BACKGROUND_PLACE,
        BUCKET_OBJECT_PROP,
        BUCKET_POSE_ACTION,
        BUCKET_LIMB_ACTION,
        BUCKET_CAMERA_COMPOSITION,
        BUCKET_LIGHTING_ENVIRONMENT,
        BUCKET_APPEARANCE_IDENTITY,
        BUCKET_CLOTHING_OUTFIT,
    ]
    enabled = [bucket for bucket in ordered if bucket in rules.general_allow_buckets]
    return ", ".join(enabled) if enabled else "(none)"


def _bucket_count_template() -> Dict[str, int]:
    return {
        BUCKET_BACKGROUND_PLACE: 0,
        BUCKET_OBJECT_PROP: 0,
        BUCKET_POSE_ACTION: 0,
        BUCKET_LIMB_ACTION: 0,
        BUCKET_CAMERA_COMPOSITION: 0,
        BUCKET_LIGHTING_ENVIRONMENT: 0,
        BUCKET_APPEARANCE_IDENTITY: 0,
        BUCKET_CLOTHING_OUTFIT: 0,
        BUCKET_UNKNOWN: 0,
    }


def classify_general_tag(tag: str) -> str:
    norm = normalize_rule_tag(tag)
    if not norm:
        return BUCKET_UNKNOWN
    if norm in EXACT_ALLOW_CAMERA:
        return BUCKET_CAMERA_COMPOSITION
    if norm in EXACT_ALLOW_LIGHTING:
        return BUCKET_LIGHTING_ENVIRONMENT
    if norm in EXACT_ALLOW_POSE:
        return BUCKET_POSE_ACTION
    if norm in EXACT_ALLOW_LIMB:
        return BUCKET_LIMB_ACTION
    if norm in EXACT_ALLOW_BACKGROUND:
        return BUCKET_BACKGROUND_PLACE
    if norm in EXACT_ALLOW_OBJECT:
        return BUCKET_OBJECT_PROP
    if norm in EXACT_DENY_CLOTHING:
        return BUCKET_CLOTHING_OUTFIT
    if norm in EXACT_DENY_APPEARANCE:
        return BUCKET_APPEARANCE_IDENTITY
    if _matches_any(norm, COMPILED_REGEX_ALLOW_CAMERA):
        return BUCKET_CAMERA_COMPOSITION
    if _matches_any(norm, COMPILED_REGEX_ALLOW_LIGHTING):
        return BUCKET_LIGHTING_ENVIRONMENT
    if _matches_any(norm, COMPILED_REGEX_ALLOW_POSE):
        return BUCKET_POSE_ACTION
    if _matches_any(norm, COMPILED_REGEX_ALLOW_LIMB):
        return BUCKET_LIMB_ACTION
    if _matches_any(norm, COMPILED_REGEX_ALLOW_BACKGROUND):
        return BUCKET_BACKGROUND_PLACE
    if _matches_any(norm, COMPILED_REGEX_ALLOW_OBJECT):
        return BUCKET_OBJECT_PROP
    if _matches_any(norm, COMPILED_REGEX_DENY_CLOTHING):
        return BUCKET_CLOTHING_OUTFIT
    if _matches_any(norm, COMPILED_REGEX_DENY_APPEARANCE):
        return BUCKET_APPEARANCE_IDENTITY
    return BUCKET_UNKNOWN


_DB_CLOTHING_HINTS = (
    "clothing",
    "outfit",
    "garment",
    "uniform",
    "dress",
    "shirt",
    "skirt",
    "jacket",
    "coat",
    "pants",
    "shorts",
    "shoes",
    "boots",
    "socks",
    "thighhigh",
    "bikini",
    "swimsuit",
    "underwear",
    "gloves",
    "hat",
    "cap",
    "ribbon",
    "necktie",
    "scarf",
)
_DB_APPEARANCE_HINTS = (
    "hair",
    "eyes",
    "skin",
    "breasts",
    "female",
    "male",
    "girl",
    "boy",
    "face",
    "freckles",
    "fang",
    "ears",
    "hairstyle",
    "eyecolor",
)
_DB_POSE_HINTS = (
    "pose",
    "posture",
    "standing",
    "sitting",
    "kneeling",
    "lying",
    "running",
    "jumping",
    "crouching",
    "leaning",
    "arms",
    "legs",
    "gesture",
    "pointing",
)
_DB_BACKGROUND_HINTS = (
    "background",
    "scenery",
    "landscape",
    "indoors",
    "outdoors",
    "room",
    "classroom",
    "office",
    "forest",
    "street",
    "building",
    "sky",
    "cloud",
    "night",
    "day",
    "river",
    "mountain",
    "beach",
    "park",
    "garden",
)
_DB_OBJECT_HINTS = (
    "object",
    "weapon",
    "prop",
    "holding",
    "chair",
    "table",
    "book",
    "cup",
    "phone",
    "sword",
    "gun",
    "staff",
    "umbrella",
    "bag",
    "car",
    "bicycle",
    "train",
    "food",
    "drink",
    "instrument",
    "tool",
)


def _text_has_any(text: str, hints: Tuple[str, ...]) -> bool:
    for hint in hints:
        if hint in text:
            return True
    return False


def _lookup_danbooru_bucket(tag: str) -> str:
    try:
        from services import danbooru_client
    except Exception:
        return BUCKET_UNKNOWN
    data = danbooru_client.lookup_tag_info(
        tag,
        include_related=True,
        include_preview=False,
    )
    if not data.get("ok") or not data.get("found"):
        return BUCKET_UNKNOWN

    info = data.get("info") or {}
    category_name = str(info.get("category_name") or "").strip().lower()
    if category_name == "character":
        return BUCKET_APPEARANCE_IDENTITY
    if category_name in {"artist", "copyright", "meta", "deprecated"}:
        return BUCKET_UNKNOWN

    text_parts = [normalize_rule_tag(tag)]
    wiki = data.get("wiki") or {}
    text_parts.append(normalize_rule_tag(wiki.get("title") or ""))
    text_parts.append(normalize_rule_tag(wiki.get("body") or ""))
    related = data.get("related") or []
    if isinstance(related, list):
        for item in related[:24]:
            text_parts.append(normalize_rule_tag(item))
    hay = " ".join(part for part in text_parts if part)

    if _text_has_any(hay, _DB_CLOTHING_HINTS):
        return BUCKET_CLOTHING_OUTFIT
    if _text_has_any(hay, _DB_APPEARANCE_HINTS):
        return BUCKET_APPEARANCE_IDENTITY
    if _text_has_any(hay, _DB_POSE_HINTS):
        return BUCKET_POSE_ACTION
    if _text_has_any(hay, _DB_BACKGROUND_HINTS):
        return BUCKET_BACKGROUND_PLACE
    if _text_has_any(hay, _DB_OBJECT_HINTS):
        return BUCKET_OBJECT_PROP
    return BUCKET_UNKNOWN


def _classify_general_tag_with_safenet(tag: str, state: Optional[DanbooruSafeNetState]) -> str:
    bucket = classify_general_tag(tag)
    if bucket != BUCKET_UNKNOWN or state is None or not state.enabled:
        return bucket
    key = normalize_rule_tag(tag)
    cached = state.cache.get(key)
    if cached is not None:
        return cached
    if state.lookups >= state.max_lookups:
        state.skipped += 1
        state.cache[key] = BUCKET_UNKNOWN
        return BUCKET_UNKNOWN
    state.lookups += 1
    try:
        resolved = _lookup_danbooru_bucket(tag)
    except Exception:
        state.errors += 1
        resolved = BUCKET_UNKNOWN
    if resolved != BUCKET_UNKNOWN:
        state.resolved += 1
    state.cache[key] = resolved
    return resolved


def filter_general_tags_selective(
    tags_with_scores: List[Tuple[str, float]],
    selective_config: SelectiveRules,
    safenet_state: Optional[DanbooruSafeNetState] = None,
    debug: bool = False,
) -> Tuple[List[Tuple[str, float]], Dict[str, Dict[str, int]]]:
    kept: List[Tuple[str, float]] = []
    kept_counts = _bucket_count_template()
    dropped_counts = _bucket_count_template()
    dropped_tags: Dict[str, List[str]] = {bucket: [] for bucket in _bucket_count_template().keys()}
    kept_tags: Dict[str, List[str]] = {bucket: [] for bucket in _bucket_count_template().keys()}
    dropped_unknown_candidates: List[Tuple[str, float]] = []
    for tag, score in tags_with_scores:
        bucket = _classify_general_tag_with_safenet(tag, safenet_state)
        if bucket == BUCKET_UNKNOWN:
            is_allowed = not selective_config.drop_unknown_general
        else:
            is_allowed = bucket in selective_config.general_allow_buckets
        if is_allowed:
            kept.append((tag, score))
            kept_counts[bucket] = kept_counts.get(bucket, 0) + 1
            kept_tags.setdefault(bucket, []).append(tag)
        else:
            dropped_counts[bucket] = dropped_counts.get(bucket, 0) + 1
            dropped_tags.setdefault(bucket, []).append(tag)
            if bucket == BUCKET_UNKNOWN:
                dropped_unknown_candidates.append((tag, score))

    unknown_fallback_kept = 0
    if selective_config.drop_unknown_general and selective_config.use_unknown_fallback:
        kept_total = len(kept)
        if kept_total < DEFAULT_SELECTIVE_MIN_KEEP_TAGS and dropped_unknown_candidates:
            need = DEFAULT_SELECTIVE_MIN_KEEP_TAGS - kept_total
            fallback_take = min(DEFAULT_SELECTIVE_UNKNOWN_FALLBACK_MAX, max(0, need))
            if fallback_take > 0:
                restore = dropped_unknown_candidates[:fallback_take]
                kept.extend(restore)
                unknown_fallback_kept = len(restore)
                kept_counts[BUCKET_UNKNOWN] = kept_counts.get(BUCKET_UNKNOWN, 0) + unknown_fallback_kept
                for tag, _score in restore:
                    kept_tags.setdefault(BUCKET_UNKNOWN, []).append(tag)
                    if tag in dropped_tags.get(BUCKET_UNKNOWN, []):
                        dropped_tags[BUCKET_UNKNOWN].remove(tag)
                dropped_counts[BUCKET_UNKNOWN] = max(
                    0,
                    dropped_counts.get(BUCKET_UNKNOWN, 0) - unknown_fallback_kept,
                )
                kept.sort(key=lambda x: x[1], reverse=True)

    details: Dict[str, Dict[str, int]] = {"kept": kept_counts, "dropped": dropped_counts}
    if debug:
        details["kept_tags"] = kept_tags
        details["dropped_tags"] = dropped_tags
        details["config"] = {
            "drop_unknown_general": 1 if selective_config.drop_unknown_general else 0,
            "unknown_fallback_kept": unknown_fallback_kept,
            "use_unknown_fallback": 1 if selective_config.use_unknown_fallback else 0,
        }
        if safenet_state is not None and safenet_state.enabled:
            details["config"]["danbooru_safenet"] = 1
    return kept, details


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


def _apply_blocked_auto_tags(auto_tags: List[str], blocked_tags: List[str]) -> Tuple[List[str], List[str]]:
    blocked_patterns = _parse_blocked_tag_patterns(blocked_tags)
    if not blocked_patterns:
        return list(auto_tags), []
    kept: List[str] = []
    removed: List[str] = []
    for tag in auto_tags:
        norm = _normalize_user_tag(tag)
        if any(fnmatch.fnmatchcase(norm, pattern) for pattern in blocked_patterns):
            removed.append(tag)
        else:
            kept.append(tag)
    return kept, _dedup_preserve(removed)


def merge_caption_tags(
    prefix_tags: List[str],
    existing_tags: List[str],
    auto_tags: List[str],
) -> List[str]:
    merged: List[str] = []
    seen = set()
    for tag in prefix_tags or []:
        norm = _normalize_user_tag(tag)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        merged.append(str(tag or "").strip())
    for tag in existing_tags or []:
        value = (tag or "").strip()
        norm = _normalize_user_tag(value)
        if not value or not norm or norm in seen:
            continue
        seen.add(norm)
        merged.append(to_caption_tag(value))
    for tag in auto_tags or []:
        norm = _normalize_user_tag(tag)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        merged.append(to_caption_tag(tag))
    return merged


def _is_literal_overwrite(write_mode: str) -> bool:
    return (write_mode or "").strip().lower() == "overwrite"


def _effective_skip_empty(write_mode: str, skip_empty: bool) -> bool:
    return bool(skip_empty and not _is_literal_overwrite(write_mode))


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
    danbooru_safenet_state: Optional[DanbooruSafeNetState] = None,
    compiled_policy: Optional[tag_policy.CompiledTagPolicy] = None,
    debug_state: Optional[Dict[str, Any]] = None,
) -> List[str]:
    general: List[Tuple[str, float]] = []
    characters: List[Tuple[str, float]] = []
    meta: List[Tuple[str, float]] = []
    artists: List[Tuple[str, float]] = []
    copyrights: List[Tuple[str, float]] = []
    ratings: List[Tuple[str, float]] = []
    unknown: List[Tuple[str, float]] = []

    categories_present = bool(categories)
    policy_drop_counts: Dict[str, int] = {}
    policy_removed: List[Tuple[str, float, str]] = []

    use_color_sanity = bool(opts.enable_color_sanity and color_presence)
    ratio_threshold = max(0.0, min(float(opts.color_ratio_threshold), 1.0))
    keep_threshold = max(0.0, min(float(opts.color_keep_if_score_ge), 1.0))

    for idx, score in enumerate(probs):
        tag = tag_policy.normalize_tag(labels[idx])
        category = categories[idx] if categories is not None and idx < len(categories) else None
        score_f = float(score)
        if compiled_policy is not None and compiled_policy.enabled:
            blocked, reason, group = compiled_policy.decision_for_index(idx)
            if blocked:
                group_key = group or reason or "policy"
                policy_drop_counts[group_key] = policy_drop_counts.get(group_key, 0) + 1
                if debug_state is not None:
                    policy_removed.append((tag, score_f, group_key))
                continue
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
    use_mcut = mode == "mcut" and not bool(getattr(opts, "simple_mode", False))
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
        opts.policy_mcut_min_general_tags
        if compiled_policy is not None and compiled_policy.enabled
        else opts.mcut_min_general_tags,
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
    wd_candidate_count = len(general + characters + meta + artists + copyrights + ratings + unknown)

    if opts.character_topk > 0 and len(characters) > opts.character_topk:
        characters = characters[: opts.character_topk]

    if opts.max_character_tags > 0 and len(characters) > opts.max_character_tags:
        characters = characters[: opts.max_character_tags]

    selective_rules = _resolve_selective_rules(opts)
    profile = selective_rules.output_profile if selective_rules else OUTPUT_PROFILE_STANDARD_FULL
    using_selective = selective_rules is not None
    selective_counts = {"kept": _bucket_count_template(), "dropped": _bucket_count_template()}
    selective_config_stats: Dict[str, int] = {}
    dropped_character_count = 0
    dropped_meta_count = 0
    dropped_artist_copyright_count = 0
    dropped_rating_count = 0

    if using_selective:
        general, selective_counts = filter_general_tags_selective(
            general,
            selective_rules,
            safenet_state=danbooru_safenet_state,
            debug=True,
        )
        selective_config_stats = selective_counts.get("config") or {}
    max_general_limit = int(getattr(opts, "max_auto_tags", 0) or 0) if getattr(opts, "simple_mode", False) else opts.max_general_tags
    if max_general_limit > 0 and len(general) > max_general_limit:
        general = general[:max_general_limit]

    split_patterns = non_character_regex
    if split_patterns is None:
        split_patterns = _compile_regex(opts.non_character_regex)
    subject_general, non_character_general = _split_general_focus(general, split_patterns)

    meta_bucket: List[Tuple[str, float]] = []
    include_meta = opts.include_meta if not using_selective else selective_rules.include_meta
    include_artist = opts.include_artist if not using_selective else selective_rules.include_artist
    include_copyright = opts.include_copyright if not using_selective else selective_rules.include_copyright
    include_character = opts.include_character if not using_selective else selective_rules.include_character
    include_rating = opts.include_rating if not using_selective else selective_rules.include_rating

    if include_meta:
        meta_bucket += meta
    if include_artist:
        meta_bucket += artists
    if include_copyright:
        meta_bucket += copyrights
    meta_bucket.sort(key=lambda x: x[1], reverse=True)
    if opts.max_meta_tags > 0 and len(meta_bucket) > opts.max_meta_tags:
        meta_bucket = meta_bucket[: opts.max_meta_tags]

    tags: List[str] = []
    emitted_general: List[Tuple[str, float]] = []
    emitted_character: List[Tuple[str, float]] = []
    emitted_meta: List[Tuple[str, float]] = []
    if using_selective:
        if opts.include_general:
            emitted_general = general
            tags.extend([t for t, _ in emitted_general])
        if include_character:
            emitted_character = characters
            tags.extend([t for t, _ in emitted_character])
        if include_meta or include_artist or include_copyright:
            emitted_meta = meta_bucket
            tags.extend([t for t, _ in emitted_meta])
        dropped_character_count = len(characters) - len(emitted_character)
        dropped_meta_count = len(meta) if not include_meta else 0
        dropped_artist_copyright_count = (len(artists) if not include_artist else 0) + (
            len(copyrights) if not include_copyright else 0
        )
        dropped_rating_count = 1 if (ratings and not include_rating) else 0
    elif focus_mode == "character":
        if opts.include_general:
            emitted_general = subject_general
            tags.extend([t for t, _ in emitted_general])
        if include_character:
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
        if include_character:
            emitted_character = characters
            tags.extend([t for t, _ in emitted_character])
        emitted_meta = meta_bucket
        tags.extend([t for t, _ in emitted_meta])

    if not categories_present and opts.include_general:
        unknown.sort(key=lambda x: x[1], reverse=True)
        tags.extend([t for t, _ in unknown])

    if getattr(opts, "simple_mode", False) and int(getattr(opts, "max_auto_tags", 0) or 0) > 0:
        tags = tags[: int(getattr(opts, "max_auto_tags", 0))]
    if opts.max_tags > 0 and len(tags) > opts.max_tags:
        tags = tags[: opts.max_tags]

    if include_rating and ratings:
        rating_tag = max(ratings, key=lambda x: x[1])[0]
        tags = [rating_tag] + tags

    if stats is not None:
        stats["general"] = stats.get("general", 0) + len(emitted_general)
        stats["character"] = stats.get("character", 0) + len(emitted_character)
        stats["meta"] = stats.get("meta", 0) + len(emitted_meta)
        stats["subject_general"] = stats.get("subject_general", 0) + len(subject_general)
        stats["non_character_general"] = stats.get("non_character_general", 0) + len(non_character_general)
        stats["rating"] = stats.get("rating", 0) + (1 if (include_rating and ratings) else 0)
        if emitted_general:
            stats["images_with_general"] = stats.get("images_with_general", 0) + 1
        if emitted_character:
            stats["images_with_character"] = stats.get("images_with_character", 0) + 1
        if using_selective:
            for bucket, count in selective_counts["kept"].items():
                key = f"selective_kept_{bucket}"
                stats[key] = stats.get(key, 0) + count
            for bucket, count in selective_counts["dropped"].items():
                key = f"selective_dropped_{bucket}"
                stats[key] = stats.get(key, 0) + count
            stats["selective_unknown_fallback_kept"] = (
                stats.get("selective_unknown_fallback_kept", 0)
                + int(selective_config_stats.get("unknown_fallback_kept", 0))
            )
            stats["selective_dropped_character"] = stats.get("selective_dropped_character", 0) + dropped_character_count
            stats["selective_dropped_meta"] = stats.get("selective_dropped_meta", 0) + dropped_meta_count
            stats["selective_dropped_artist_copyright"] = (
                stats.get("selective_dropped_artist_copyright", 0)
                + dropped_artist_copyright_count
            )
            stats["selective_dropped_rating"] = stats.get("selective_dropped_rating", 0) + dropped_rating_count
            stats["selective_dropped_meta_rating"] = (
                stats.get("selective_dropped_meta_rating", 0)
                + dropped_meta_count
                + dropped_rating_count
            )

    out: List[str] = []
    seen = set()
    for tag in tags:
        fmt = normalize_rule_tag(tag) if getattr(opts, "simple_mode", False) else _format_tag(tag, opts.replace_underscore)
        if fmt not in seen:
            out.append(fmt)
            seen.add(fmt)

    out = _apply_excludes(out, exclude_tags, exclude_regex)
    if compiled_policy is not None and compiled_policy.enabled:
        filtered: List[str] = []
        for tag in out:
            blocked, reason, group = compiled_policy.decision_for_tag(tag)
            if blocked:
                group_key = group or reason or "policy"
                policy_drop_counts[group_key] = policy_drop_counts.get(group_key, 0) + 1
                if debug_state is not None:
                    policy_removed.append((tag, 0.0, group_key))
                continue
            filtered.append(tag)
        out = filtered
    if stats is not None and policy_drop_counts:
        stats["policy_drops"] = stats.get("policy_drops", 0) + sum(policy_drop_counts.values())
        stats["policy_images_with_drops"] = stats.get("policy_images_with_drops", 0) + 1
        for group, count in policy_drop_counts.items():
            stats[f"policy_drop_{group}"] = stats.get(f"policy_drop_{group}", 0) + count
    if debug_state is not None:
        debug_state["output_profile"] = profile
        debug_state["wd_candidates"] = wd_candidate_count
        if using_selective:
            debug_state["enabled_buckets"] = sorted(selective_rules.general_allow_buckets)
            debug_state["drop_unknown_general"] = selective_rules.drop_unknown_general
            debug_state["kept_counts"] = dict(selective_counts["kept"])
            debug_state["dropped_counts"] = dict(selective_counts["dropped"])
            debug_state["kept_tags"] = dict(selective_counts.get("kept_tags") or {})
            debug_state["dropped_tags"] = dict(selective_counts.get("dropped_tags") or {})
            debug_state["unknown_fallback_kept"] = int(selective_config_stats.get("unknown_fallback_kept", 0))
            debug_state["dropped_character"] = dropped_character_count
            debug_state["dropped_meta"] = dropped_meta_count
            debug_state["dropped_artist_copyright"] = dropped_artist_copyright_count
            debug_state["dropped_rating"] = dropped_rating_count
            debug_state["dropped_meta_rating"] = dropped_meta_count + dropped_rating_count
        else:
            debug_state["enabled_buckets"] = []
            debug_state["drop_unknown_general"] = False
            debug_state["kept_counts"] = _bucket_count_template()
            debug_state["dropped_counts"] = _bucket_count_template()
            debug_state["unknown_fallback_kept"] = 0
            debug_state["dropped_character"] = 0
            debug_state["dropped_meta"] = 0
            debug_state["dropped_artist_copyright"] = 0
            debug_state["dropped_rating"] = 0
            debug_state["dropped_meta_rating"] = 0
        debug_state["policy_removed"] = [
            {"tag": tag, "score": score, "reason": reason}
            for tag, score, reason in policy_removed
        ]
        debug_state["policy_drop_counts"] = dict(policy_drop_counts)
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
    protected_literals: Optional[List[str]] = None,
) -> str:
    try:
        from services import normalizer

        tf = normalizer.TagFile(
            path=Path(""),
            main=main,
            optional=optional,
            warning=warning,
            protected_literals=list(protected_literals or []),
        )
        text = normalizer.format_tag_file(tf)
    except Exception:
        text = join_tags(main)
    return _apply_output_formatting(text, newline_end, strip_whitespace)


def _caption_backup_path(dataset_path: Path, caption_path: Path, timestamp: str) -> Path:
    try:
        rel = caption_path.resolve().relative_to(dataset_path.resolve())
    except Exception:
        rel = Path(caption_path.name)
    folder = timestamp if str(timestamp).startswith("offline_tagger_") else f"offline_tagger_{timestamp}"
    return dataset_path / ".batchbench_backup" / folder / rel


def _prepare_caption_backup(
    dataset_path: Path,
    caption_paths: List[Path],
    timestamp: str,
) -> Tuple[Path, int]:
    backup_root = dataset_path / ".batchbench_backup" / f"offline_tagger_{timestamp}"
    for caption_path in caption_paths:
        if not caption_path.exists():
            continue
        backup_path = _caption_backup_path(dataset_path, caption_path, timestamp)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(caption_path, backup_path)
    return backup_root, sum(1 for path in caption_paths if path.exists())


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _write_caption_safely(
    dataset_path: Path,
    caption_path: Path,
    text: str,
    *,
    backup_existing: bool,
    timestamp: str,
) -> Tuple[bool, bool, Optional[Path]]:
    old_text: Optional[str] = None
    if caption_path.exists():
        old_text = caption_path.read_text(encoding="utf-8")
        if old_text == text:
            return False, False, None

    backup_path: Optional[Path] = None
    backup_created = False
    if old_text is not None and backup_existing:
        backup_path = _caption_backup_path(dataset_path, caption_path, timestamp)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(caption_path, backup_path)
        backup_created = True

    _write_text_atomic(caption_path, text)
    return True, backup_created, backup_path


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


def _sum_buckets(counts: Dict[str, int], keys: List[str]) -> int:
    total = 0
    for key in keys:
        total += int(counts.get(key, 0))
    return total


def _format_preview_reason(debug_state: Dict[str, Any]) -> str:
    kept = debug_state.get("kept_counts") or {}
    dropped = debug_state.get("dropped_counts") or {}
    kept_bg = int(kept.get(BUCKET_BACKGROUND_PLACE, 0))
    kept_obj = int(kept.get(BUCKET_OBJECT_PROP, 0))
    kept_pose = _sum_buckets(kept, [BUCKET_POSE_ACTION, BUCKET_LIMB_ACTION])
    kept_camera = int(kept.get(BUCKET_CAMERA_COMPOSITION, 0))
    kept_light = int(kept.get(BUCKET_LIGHTING_ENVIRONMENT, 0))
    drop_clothes = int(dropped.get(BUCKET_CLOTHING_OUTFIT, 0))
    drop_appearance = int(dropped.get(BUCKET_APPEARANCE_IDENTITY, 0))
    drop_unknown = int(dropped.get(BUCKET_UNKNOWN, 0))
    rescued_unknown = int(debug_state.get("unknown_fallback_kept", 0))
    rescue_text = f" fallback_unknown={rescued_unknown}" if rescued_unknown > 0 else ""
    return (
        f"kept(bg={kept_bg},obj={kept_obj},pose={kept_pose},camera={kept_camera},lighting={kept_light}) "
        f"dropped(clothes={drop_clothes},appearance={drop_appearance},unknown={drop_unknown})"
        f"{rescue_text}"
    )


def _active_cleanup_filters(opts: TaggerOptions, rules: Optional[SelectiveRules]) -> List[str]:
    labels: List[str] = []
    if rules is not None and BUCKET_APPEARANCE_IDENTITY not in rules.general_allow_buckets:
        labels.append("appearance and identity tags")
    if rules is not None and BUCKET_CLOTHING_OUTFIT not in rules.general_allow_buckets:
        labels.append("outfit and accessory tags")

    include_character = opts.include_character if rules is None else rules.include_character
    include_rating = opts.include_rating if rules is None else rules.include_rating
    include_meta = opts.include_meta if rules is None else rules.include_meta
    include_artist = opts.include_artist if rules is None else rules.include_artist
    include_copyright = opts.include_copyright if rules is None else rules.include_copyright

    if not include_character:
        labels.append("character names")
    if not include_artist or not include_copyright:
        labels.append("artist and copyright tags")
    if not include_rating or not include_meta:
        labels.append("rating and meta tags")
    if rules is not None and rules.drop_unknown_general:
        labels.append("unclassified tags")
    return labels


def _format_active_cleanup_summary(opts: TaggerOptions, rules: Optional[SelectiveRules]) -> List[str]:
    active = _active_cleanup_filters(opts, rules)
    lines = ["Active cleanup filters:"]
    if not active:
        lines.append("none")
        lines.append("Output behavior: all WD labels above confidence threshold are kept, except blacklist matches.")
        return lines
    lines.extend(f"* {label}" for label in active)
    lines.append("Output behavior:")
    lines.append("* Newly predicted " + " and ".join(active) + " will be removed.")
    lines.append("* Existing caption tags and prefix tags remain untouched.")
    return lines


def run_tagger(
    opts: TaggerOptions,
    deprecated_keys: Optional[List[str]] = None,
    progress_callback=None,
) -> Tuple[bool, List[str]]:
    lines: List[str] = []
    if deprecated_keys:
        lines.append(f"Ignored deprecated options: {', '.join(deprecated_keys)}")

    if not opts.image_exts:
        opts.image_exts = list(DEFAULT_IMAGE_EXTS)
    if opts.write_mode == "skip_if_exists":
        opts.write_mode = "skip"
    if opts.write_mode not in {"overwrite", "append", "skip"}:
        opts.write_mode = "overwrite"
    if opts.write_mode != "append":
        opts.keep_existing_tags = False
    opts.replace_existing_captions = opts.write_mode == "overwrite"
    opts.tag_policy = (
        getattr(opts, "tag_policy", tag_policy.DEFAULT_TAG_POLICY) or tag_policy.DEFAULT_TAG_POLICY
    ).strip().lower()
    if opts.tag_policy not in tag_policy.PROFILES:
        opts.tag_policy = tag_policy.DEFAULT_TAG_POLICY
    opts.policy_keep_tags = _parse_normalized_tag_list(getattr(opts, "policy_keep_tags", []) or [])
    opts.policy_block_tags = _parse_normalized_tag_list(getattr(opts, "policy_block_tags", []) or [])
    opts.policy_block_regex = _parse_regex_list(getattr(opts, "policy_block_regex", []) or [])
    opts.block_permanent_marks = bool(getattr(opts, "block_permanent_marks", False))
    if (opts.threshold_mode or "").strip().lower() not in {"fixed", "mcut"}:
        opts.threshold_mode = DEFAULT_THRESHOLD_MODE
    opts.output_profile = _resolve_output_profile(opts.output_profile)
    opts.simple_mode = bool(getattr(opts, "simple_mode", False))
    opts.tag_strength = max(
        0,
        min(_parse_int(getattr(opts, "tag_strength", DEFAULT_TAG_STRENGTH), DEFAULT_TAG_STRENGTH), 100),
    )
    opts.max_auto_tags = max(
        1,
        min(_parse_int(getattr(opts, "max_auto_tags", DEFAULT_MAX_AUTO_TAGS), DEFAULT_MAX_AUTO_TAGS), 100),
    )
    opts.prefix_tags = _parse_normalized_tag_list(getattr(opts, "prefix_tags", []) or [])
    opts.blocked_tags = _parse_blocked_tag_patterns(getattr(opts, "blocked_tags", []) or [])
    opts.backup_existing = bool(getattr(opts, "backup_existing", True))
    if opts.write_mode == "overwrite":
        opts.backup_existing = True
    if opts.simple_mode:
        opts.threshold_mode = "fixed"
        opts.general_threshold = max(0.01, min(float(opts.general_threshold), 0.99))
        opts.include_general = True
        opts.danbooru_safenet = False
        opts.dedupe = True
        opts.sort_tags = False
        opts.max_general_tags = opts.max_auto_tags
    if (opts.tag_focus_mode or "").strip().lower() not in {"all", "character", "non_character"}:
        opts.tag_focus_mode = DEFAULT_TAG_FOCUS_MODE
    if (opts.backend or "").strip().lower() not in {"transformers", "onnx"}:
        opts.backend = DEFAULT_BACKEND
    opts.min_threshold_floor = max(0.0, min(float(opts.min_threshold_floor), 1.0))
    opts.mcut_relax_general = max(0.0, min(float(opts.mcut_relax_general), 1.0))
    opts.mcut_relax_character = max(0.0, min(float(opts.mcut_relax_character), 1.0))
    opts.mcut_relax_meta = max(0.0, min(float(opts.mcut_relax_meta), 1.0))
    opts.mcut_min_general_tags = max(0, int(opts.mcut_min_general_tags))
    opts.policy_mcut_min_general_tags = max(
        0,
        int(
            getattr(
                opts,
                "policy_mcut_min_general_tags",
                DEFAULT_POLICY_MCUT_MIN_GENERAL_TAGS
                if opts.tag_policy == tag_policy.POLICY_CHARACTER_IDENTITY_OMITTED
                else opts.mcut_min_general_tags,
            )
        ),
    )
    opts.mcut_min_character_tags = max(0, int(opts.mcut_min_character_tags))
    opts.mcut_min_meta_tags = max(0, int(opts.mcut_min_meta_tags))
    opts.trigger_tag = (opts.trigger_tag or "").strip()
    opts.color_ratio_threshold = max(0.0, min(float(opts.color_ratio_threshold), 1.0))
    opts.color_min_saturation = max(0.0, min(float(opts.color_min_saturation), 1.0))
    opts.color_min_value = max(0.0, min(float(opts.color_min_value), 1.0))
    opts.color_keep_if_score_ge = max(0.0, min(float(opts.color_keep_if_score_ge), 1.0))
    opts.color_downscale = max(16, int(opts.color_downscale or DEFAULT_COLOR_DOWNSCALE))
    opts.selective_keep_background_place = bool(opts.selective_keep_background_place)
    opts.selective_keep_object_prop = bool(opts.selective_keep_object_prop)
    opts.selective_keep_pose_action = bool(opts.selective_keep_pose_action)
    opts.selective_keep_appearance = bool(opts.selective_keep_appearance)
    opts.selective_keep_clothing = bool(opts.selective_keep_clothing)
    opts.selective_keep_character_names = bool(opts.selective_keep_character_names)
    opts.selective_keep_artist_copyright = bool(getattr(opts, "selective_keep_artist_copyright", False))
    opts.selective_keep_rating_meta = bool(opts.selective_keep_rating_meta)
    opts.selective_keep_unknown_general = bool(opts.selective_keep_unknown_general)
    opts.danbooru_safenet = bool(opts.danbooru_safenet)
    selective_rules = _resolve_selective_rules(opts)
    effective_profile = selective_rules.output_profile if selective_rules else OUTPUT_PROFILE_STANDARD_FULL
    danbooru_safenet_state = DanbooruSafeNetState(
        enabled=bool(opts.danbooru_safenet and selective_rules is not None),
        max_lookups=DEFAULT_DANBOORU_SAFENET_MAX_LOOKUPS,
    )

    worker_result = _run_external_worker(opts, deprecated_keys)
    if worker_result is not None:
        return worker_result
    if getattr(sys, "frozen", False) and os.environ.get(_WORKER_ENV_FLAG) != "1":
        roots = ", ".join(str(root) for root in _candidate_source_roots()) or "(none)"
        lines.append(
            "Offline Tagger in the EXE build runs through an external Python worker "
            "because PyTorch native DLLs are not reliable inside the frozen process."
        )
        lines.append("No usable worker Python was found.")
        lines.append("Expected one of: .venv\\Scripts\\python.exe or venv\\Scripts\\python.exe in the BatchBench source folder.")
        lines.append("Set BATCHBENCH_SOURCE_ROOT to the BatchBench source folder or BATCHBENCH_OFFLINE_TAGGER_PYTHON to a Python that can import torch and transformers.")
        lines.append(f"Detected source roots: {roots}")
        return False, lines

    if not opts.dataset_path.exists() or not opts.dataset_path.is_dir():
        lines.append(f"Dataset folder not found: {opts.dataset_path}")
        return False, lines

    lines.append(f"Dataset: {opts.dataset_path}")
    lines.append(f"Model: {opts.model_id}")
    active_profile = tag_policy.get_profile(opts.tag_policy)
    lines.append(f"Tag policy: {active_profile.label}")
    lines.append(f"Policy version: {active_profile.version}")
    lines.append(
        "Blocked groups: "
        + (", ".join(sorted(active_profile.blocked_groups)) if active_profile.blocked_groups else "none")
    )
    lines.append(f"Permanent marks: {'on' if opts.block_permanent_marks else 'off'}")
    lines.append(
        "Existing captions: "
        + ("replace" if opts.write_mode == "overwrite" else "skip" if opts.write_mode == "skip" else "append (legacy)")
    )
    lines.append(f"Preview only: {'on' if opts.preview_only else 'off'}")
    if opts.write_mode == "append":
        lines.append("[WARN] Legacy append write_mode is active; new Offline Tagger runs default to replacement.")
    wd_fix = bool(opts.force_wd_bgr_fix) and _is_wd_family(opts.model_id)
    lines.append(f"WD color fix {'ON (RGB->BGR)' if wd_fix else 'OFF'}")
    if opts.simple_mode:
        lines.append("Mode: Auto Tag Assist")
        lines.append(f"Minimum confidence: {opts.general_threshold:.2f}")
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
        if opts.tag_policy == tag_policy.POLICY_CHARACTER_IDENTITY_OMITTED:
            lines.append(f"Policy MCUT minimum general tags: {opts.policy_mcut_min_general_tags}")
    lines.append(
        f"Output mode: {'skip_if_exists' if opts.write_mode == 'skip' else opts.write_mode}"
    )
    lines.append(f"Output profile: {effective_profile}")
    if selective_rules:
        lines.append(f"Enabled buckets: {_enabled_bucket_labels(selective_rules)}")
        lines.append(f"Drop unknown general tags: {'yes' if selective_rules.drop_unknown_general else 'no'}")
        if selective_rules.drop_unknown_general and selective_rules.use_unknown_fallback:
            lines.append(
                "Sparse fallback: "
                f"restore top unknown tags when kept<{DEFAULT_SELECTIVE_MIN_KEEP_TAGS} "
                f"(max_restore={DEFAULT_SELECTIVE_UNKNOWN_FALLBACK_MAX})"
            )
        lines.append(f"Danbooru safe-net: {'on' if danbooru_safenet_state.enabled else 'off'}")
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
    compiled_policy = tag_policy.compile_policy(
        opts.tag_policy,
        labels,
        categories,
        category_ids.character,
        custom_keep=opts.policy_keep_tags,
        custom_block=opts.policy_block_tags,
        custom_block_regex=opts.policy_block_regex,
        permanent_marks=opts.block_permanent_marks,
    )
    if compiled_policy.enabled:
        lines.append(
            f"Policy compiled: {sum(1 for blocked in compiled_policy.blocked_mask if blocked)} "
            f"blocked label(s) of {len(compiled_policy.blocked_mask)}."
        )
    effective_include_character = opts.include_character if not selective_rules else selective_rules.include_character
    effective_include_rating = opts.include_rating if not selective_rules else selective_rules.include_rating
    effective_meta_scope = (
        f"{'on' if opts.include_meta else 'off'} / "
        f"{'on' if opts.include_artist else 'off'} / "
        f"{'on' if opts.include_copyright else 'off'}"
    )
    if selective_rules:
        effective_meta_scope = (
            f"{'on' if selective_rules.include_meta else 'off'} / "
            f"{'on' if selective_rules.include_artist else 'off'} / "
            f"{'on' if selective_rules.include_copyright else 'off'}"
        )
    lines.append(f"Include general tags: {'on' if opts.include_general else 'off'}")
    lines.append(f"Include character tags: {'on' if effective_include_character else 'off'}")
    lines.append(f"Include rating tags: {'on' if effective_include_rating else 'off'}")
    if selective_rules:
        lines.append("Tag focus mode: auto (profile managed)")
    else:
        lines.append(f"Tag focus mode: {opts.tag_focus_mode}")
    lines.append(f"Include meta/artist/copyright: {effective_meta_scope}")
    lines.append(
        f"Threshold floor: {opts.min_threshold_floor} (policy)"
    )
    if opts.simple_mode:
        lines.append(f"Prefix tags: {', '.join(opts.prefix_tags) if opts.prefix_tags else 'none'}")
        lines.append(f"Blocked auto tags: {len(opts.blocked_tags)}")
        lines.append(f"Maximum automatic tags per image: {opts.max_auto_tags}")
        lines.append(f"Caption backups: {'on' if opts.backup_existing else 'off'}")
    else:
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

    backup_timestamp = time.strftime("%Y%m%d_%H%M%S")
    prebacked_caption_count = 0
    if opts.write_mode == "overwrite" and opts.backup_existing and not opts.preview_only:
        caption_paths = [path.with_suffix(".txt") for path in paths if path.with_suffix(".txt").exists()]
        if caption_paths:
            try:
                backup_root, prebacked_caption_count = _prepare_caption_backup(
                    opts.dataset_path,
                    caption_paths,
                    backup_timestamp,
                )
            except Exception as exc:
                lines.append(f"[ERROR] Caption backup failed before writing: {exc}")
                return False, lines
            lines.append(f"Caption backup: {backup_root}")
            lines.append(f"Caption files backed up: {prebacked_caption_count}")

    processed = 0
    written = 0
    no_change = 0
    missing_created = 0
    backups_created = prebacked_caption_count
    auto_tags_added = 0
    blocked_by_user_total = 0
    filtered_semantic_total = 0
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
    final_policy_leaks = 0
    policy_leak_details: List[str] = []

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
            if progress_callback is not None:
                try:
                    progress_callback(processed, total_images, force)
                except Exception:
                    pass
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
            image_debug: Dict[str, Any] = {}
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
                danbooru_safenet_state=danbooru_safenet_state,
                compiled_policy=compiled_policy,
                debug_state=image_debug,
            )
            if opts.debug_color_sanity and color_debug:
                color_drop_total += len(color_debug)
                color_drop_images += 1
            txt_path = path.with_suffix(".txt")
            existing_main: List[str] = []
            existing_optional: List[str] = []
            existing_warning: Optional[str] = None
            if txt_path.exists():
                existing_main, existing_optional, existing_warning = _read_tag_file(txt_path)

            blocked_auto_tags: List[str] = []
            auto_tags = list(tags)
            if opts.simple_mode:
                auto_tags, blocked_auto_tags = _apply_blocked_auto_tags(auto_tags, opts.blocked_tags)
                blocked_by_user_total += len(blocked_auto_tags)
                dropped_counts = image_debug.get("dropped_counts") or {}
                filtered_semantic_total += (
                    int(dropped_counts.get(BUCKET_APPEARANCE_IDENTITY, 0))
                    + int(dropped_counts.get(BUCKET_CLOTHING_OUTFIT, 0))
                    + int(dropped_counts.get(BUCKET_UNKNOWN, 0))
                    + int(image_debug.get("dropped_character", 0))
                    + int(image_debug.get("dropped_artist_copyright", 0))
                    + int(image_debug.get("dropped_meta_rating", 0))
                )
                existing_for_merge = existing_main if opts.write_mode == "append" and opts.keep_existing_tags else []
                merged_main = merge_caption_tags(opts.prefix_tags, existing_for_merge, auto_tags)
            else:
                auto_tags = _apply_trigger_tag(auto_tags, opts.trigger_tag)
                skip_empty_effective = _effective_skip_empty(opts.write_mode, opts.skip_empty)
                if skip_empty_effective and not auto_tags:
                    processed += 1
                    _report_progress()
                    continue
                if opts.write_mode == "overwrite":
                    merged_main = auto_tags
                else:
                    if opts.keep_existing_tags:
                        merged_main = list(existing_main) + auto_tags
                    else:
                        merged_main = auto_tags
                if opts.dedupe:
                    merged_main = _dedup_preserve(merged_main)
                if opts.sort_tags:
                    merged_main = sorted(merged_main)
                merged_main = _apply_trigger_tag(merged_main, opts.trigger_tag)

            protected_literals = [opts.trigger_tag] if opts.trigger_tag else []
            if opts.simple_mode:
                protected_literals.extend(opts.prefix_tags or [])
            auto_tags = normalize_caption_tags(auto_tags)
            merged_main = normalize_caption_tags(
                merged_main,
                protected_literals=protected_literals,
            )

            audit_trigger = opts.trigger_tag or (opts.prefix_tags[0] if opts.prefix_tags else "")
            leaks = tag_policy.audit_tags(merged_main, compiled_policy, trigger_tag=audit_trigger)
            if leaks:
                final_policy_leaks += len(leaks)
                rel_name = path.relative_to(opts.dataset_path).as_posix() if path.is_relative_to(opts.dataset_path) else path.name
                for leak_tag, reason, _group in leaks[:10]:
                    policy_leak_details.append(f"{rel_name}: {leak_tag} [{reason}]")

            skip_empty_effective = _effective_skip_empty(opts.write_mode, opts.skip_empty)
            if opts.simple_mode and skip_empty_effective and not merged_main and not opts.preview_only:
                processed += 1
                _report_progress()
                continue

            if opts.preview_limit > 0 and sample_count < opts.preview_limit:
                if opts.simple_mode:
                    dropped_counts = image_debug.get("dropped_counts") or {}
                    rel_name = path.relative_to(opts.dataset_path).as_posix() if path.is_relative_to(opts.dataset_path) else path.name
                    samples.append(f"[PREVIEW] {rel_name}")
                    samples.append(f"threshold: {opts.general_threshold:.2f}")
                    samples.append(f"WD candidates above threshold: {int(image_debug.get('wd_candidates', len(auto_tags)))}")
                    samples.append(
                        f"removed by appearance/identity cleanup: "
                        f"{int(dropped_counts.get(BUCKET_APPEARANCE_IDENTITY, 0))}"
                    )
                    samples.append(
                        f"removed by outfit cleanup: "
                        f"{int(dropped_counts.get(BUCKET_CLOTHING_OUTFIT, 0))}"
                    )
                    samples.append(f"removed by character-name cleanup: {int(image_debug.get('dropped_character', 0))}")
                    samples.append(
                        f"removed by artist/copyright cleanup: "
                        f"{int(image_debug.get('dropped_artist_copyright', 0))}"
                    )
                    samples.append(f"removed by rating/meta cleanup: {int(image_debug.get('dropped_meta_rating', 0))}")
                    samples.append(f"removed by unclassified cleanup: {int(dropped_counts.get(BUCKET_UNKNOWN, 0))}")
                    samples.append(f"removed by blacklist: {len(blocked_auto_tags)}")
                    samples.append(f"final automatic tags: {len(auto_tags)}")
                    samples.append(f"automatic tags after cleanup: {', '.join(auto_tags) if auto_tags else '(none)'}")
                    if opts.write_mode == "append":
                        samples.append(f"existing caption tags kept: {len(existing_main)}")
                    samples.append(f"caption after merge: {', '.join(merged_main) if merged_main else '(empty)'}")
                elif opts.preview_only and image_debug.get("output_profile") != OUTPUT_PROFILE_STANDARD_FULL:
                    samples.append(path.name)
                    samples.append(_format_preview_reason(image_debug))
                    samples.append(f"final: {', '.join(auto_tags)}")
                else:
                    samples.append(f"[{sample_count + 1}/{opts.preview_limit}] {path.name}")
                    samples.append("Kept:")
                    samples.append(f"  {', '.join(auto_tags) if auto_tags else '(none)'}")
                    removed = image_debug.get("policy_removed") or []
                    if removed:
                        samples.append("Removed by policy:")
                        for item in removed[:25]:
                            samples.append(
                                f"  {item.get('tag')} {float(item.get('score', 0.0)):.3f} [{item.get('reason')}]"
                            )
                if opts.debug_color_sanity and color_debug:
                    for item in color_debug:
                        samples.append(f"  [color_sanity] dropped {item}")
                sample_count += 1

            if opts.preview_only:
                processed += 1
                _report_progress()
                continue

            if merged_main or not skip_empty_effective:
                text = _format_tag_file(
                    merged_main,
                    existing_optional if opts.write_mode in {"append", "overwrite"} else [],
                    existing_warning if opts.write_mode in {"append", "overwrite"} else None,
                    opts.newline_end,
                    opts.strip_whitespace,
                    protected_literals=protected_literals,
                )
                existed_before = txt_path.exists()
                try:
                    changed, backup_created, _backup_path = _write_caption_safely(
                        opts.dataset_path,
                        txt_path,
                        text,
                        backup_existing=opts.backup_existing
                        and not (opts.write_mode == "overwrite" and prebacked_caption_count > 0),
                        timestamp=backup_timestamp,
                    )
                    if changed:
                        written += 1
                        if not existed_before:
                            missing_created += 1
                        if backup_created:
                            backups_created += 1
                        if opts.simple_mode:
                            existing_keys = {
                                _normalize_user_tag(tag)
                                for tag in (opts.prefix_tags + (existing_main if opts.write_mode == "append" else []))
                            }
                            auto_tags_added += sum(
                                1 for tag in auto_tags if _normalize_user_tag(tag) not in existing_keys
                            )
                            if opts.preview_limit > 0 and sample_count <= opts.preview_limit:
                                samples.append(f"[OK] {path.name}")
                                samples.append(f"  existing tags kept: {len(existing_main) if opts.write_mode == 'append' else 0}")
                                samples.append(f"  automatic tags added: {len(auto_tags)}")
                                samples.append(f"  backup: {'created' if backup_created else 'not needed'}")
                                samples.append("  caption: updated")
                    else:
                        no_change += 1
                except Exception as exc:
                    errors += 1
                    lines.append(f"[ERROR] {path.name}: failed to write caption: {exc}")

            processed += 1
            _report_progress()

    _report_progress(force=True)

    if samples:
        if opts.simple_mode:
            lines.extend(_format_active_cleanup_summary(opts, selective_rules))
            lines.append("Sample results:")
        else:
            lines.append("Sample tags:")
        lines.extend(samples)

    if tag_stats and not opts.simple_mode:
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
        if selective_rules:
            lines.append(
                "Selective kept counts: "
                f"background_place={tag_stats.get('selective_kept_background_place', 0)}, "
                f"object_prop={tag_stats.get('selective_kept_object_prop', 0)}, "
                f"pose_action={tag_stats.get('selective_kept_pose_action', 0)}, "
                f"limb_action={tag_stats.get('selective_kept_limb_action', 0)}"
            )
            lines.append(
                "Selective dropped counts: "
                f"appearance_identity={tag_stats.get('selective_dropped_appearance_identity', 0)}, "
                f"clothing_outfit={tag_stats.get('selective_dropped_clothing_outfit', 0)}, "
                f"unknown={tag_stats.get('selective_dropped_unknown', 0)}"
            )
            lines.append(
                f"Selective unknown fallback kept: {tag_stats.get('selective_unknown_fallback_kept', 0)}"
            )
            lines.append(f"Selective dropped character tags: {tag_stats.get('selective_dropped_character', 0)}")
            lines.append(
                f"Selective dropped artist/copyright tags: {tag_stats.get('selective_dropped_artist_copyright', 0)}"
            )
            lines.append(
                f"Selective dropped meta/rating tags: {tag_stats.get('selective_dropped_meta_rating', 0)}"
            )
    if tag_stats.get("policy_drops", 0) or compiled_policy.enabled:
        lines.append(f"Policy drops: {tag_stats.get('policy_drops', 0)}")
        for group in sorted(
            key.replace("policy_drop_", "")
            for key in tag_stats.keys()
            if key.startswith("policy_drop_")
        ):
            lines.append(f"  {group}: {tag_stats.get('policy_drop_' + group, 0)}")
        lines.append(f"Images with policy drops: {tag_stats.get('policy_images_with_drops', 0)}")
    if policy_leak_details:
        lines.append("Final policy leak details:")
        lines.extend(policy_leak_details[:50])
    lines.append(f"Final policy leaks: {final_policy_leaks}")
    if opts.debug_color_sanity:
        lines.append(
            f"Color sanity drops: {color_drop_total} tags across {color_drop_images} images."
        )
    if danbooru_safenet_state.enabled:
        lines.append(
            "Danbooru safe-net stats: "
            f"lookups={danbooru_safenet_state.lookups}, "
            f"resolved={danbooru_safenet_state.resolved}, "
            f"errors={danbooru_safenet_state.errors}, "
            f"lookup_cap_skips={danbooru_safenet_state.skipped}"
        )

    if opts.preview_only:
        if opts.simple_mode:
            lines.append("Auto Tag Assist preview complete")
            lines.append(f"- Images scanned: {processed}")
            lines.append("- Captions changed: 0")
            lines.append("- Backups created: 0")
            lines.append(f"- Blocked by user blacklist: {blocked_by_user_total}")
            lines.append(f"- Filtered by cleanup options: {filtered_semantic_total}")
            lines.append(f"- Errors: {errors}")
            lines.append("Preview mode made no filesystem changes.")
        else:
            lines.append(f"Preview done. Processed {processed} images (no files written).")
    else:
        if opts.simple_mode:
            lines.append("Auto Tag Assist complete")
            lines.append(f"- Images scanned: {processed}")
            lines.append(f"- Captions changed: {written}")
            lines.append(f"- No-change captions: {no_change}")
            lines.append(f"- Missing captions created: {missing_created}")
            lines.append(f"- Backups created: {backups_created}")
            lines.append(f"- Auto tags added: {auto_tags_added}")
            lines.append(f"- Blocked by user blacklist: {blocked_by_user_total}")
            lines.append(f"- Filtered by cleanup options: {filtered_semantic_total}")
            lines.append(f"- Errors: {errors}")
        else:
            lines.append(f"Processed: {processed}")
            lines.append(f"Written: {written}")
            lines.append(f"Skipped existing: {skipped}")
            lines.append(f"Backed up: {backups_created}")
            lines.append(
                "Generated tags kept: "
                f"{tag_stats.get('general', 0) + tag_stats.get('character', 0) + tag_stats.get('meta', 0) + tag_stats.get('rating', 0)}"
            )
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

    if isinstance(form, dict):
        form_opts = dict(form)
    else:
        form_opts = {}
        for key in form.keys():
            values = form.getlist(key) if hasattr(form, "getlist") else []
            form_opts[key] = values[-1] if values else form.get(key)
    form_opts["input_dir"] = raw_folder
    for key in (
        "recursive",
        "replace_existing_captions",
        "include_character",
        "include_rating",
        "include_meta",
        "include_copyright",
        "include_artist",
        "enable_color_sanity",
        "block_permanent_marks",
        "selective_keep_background_place",
        "selective_keep_object_prop",
        "selective_keep_pose_action",
        "selective_keep_appearance",
        "selective_keep_clothing",
        "selective_keep_character_names",
        "selective_keep_artist_copyright",
        "selective_keep_rating_meta",
        "selective_keep_unknown_general",
        "remove_appearance_identity",
        "remove_outfit_accessory",
        "remove_character_names",
        "remove_artist_copyright",
        "remove_rating_meta",
        "remove_unclassified",
        "danbooru_safenet",
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
    presence = (ctx or {}).get("discord_presence") if isinstance(ctx, dict) else None

    def _presence_progress(processed, total, force=False):
        if not presence or not hasattr(presence, "report_activity"):
            return
        phase = "preview" if opts.preview_only else ("complete" if force and processed >= total else "running")
        presence.report_activity("offline", phase=phase, current=processed, total=total)

    ok, lines = run_tagger(opts, deprecated_keys=deprecated, progress_callback=_presence_progress)
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
    parser.add_argument("--run-options", help="Run tagger from a JSON options payload.")
    parser.add_argument("--result", help="Write JSON result for --run-options.")
    args = parser.parse_args()

    if args.run_options:
        if not args.result:
            print("--result is required with --run-options", file=sys.stderr)
            sys.exit(2)
        payload = json.loads(Path(args.run_options).read_text(encoding="utf-8"))
        opts = _tagger_options_from_payload(payload.get("options") or {})
        deprecated = list(payload.get("deprecated_keys") or [])
        ok, lines = run_tagger(opts, deprecated_keys=deprecated)
        Path(args.result).write_text(
            json.dumps({"ok": ok, "lines": lines}, ensure_ascii=False),
            encoding="utf-8",
        )
        sys.exit(0 if ok else 1)

    if args.dump:
        print(TAGGER_POLICY)
        sys.exit(0)
    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    _cli()
