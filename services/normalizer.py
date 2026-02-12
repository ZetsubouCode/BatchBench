import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from utils.io import readable_path

# Default image extensions we consider when pairing txt files
DEFAULT_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


@dataclass
class TagFile:
    path: Path
    main: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)
    warning: Optional[str] = None

    def clone(self) -> "TagFile":
        return TagFile(
            path=self.path,
            main=list(self.main),
            optional=list(self.optional),
            warning=self.warning,
        )


@dataclass
class NormalizeOptions:
    dataset_path: Path
    recursive: bool = False
    include_missing_txt: bool = False
    preset_type: str = ""
    preset_file: str = ""
    extra_remove: Set[str] = field(default_factory=set)
    extra_keep: Set[str] = field(default_factory=set)
    move_unknown_background_to_optional: bool = False
    background_threshold: Optional[float] = None
    normalize_order: bool = True
    preview_limit: int = 30
    backup_enabled: bool = True
    image_exts: List[str] = field(default_factory=lambda: list(DEFAULT_IMAGE_EXTS))
    identity_tags: Set[str] = field(default_factory=set)
    pinned_tags: List[str] = field(default_factory=list)


def clean_input_list(raw: str) -> Set[str]:
    """
    Accepts comma/newline separated strings and returns a clean set of tags.
    """
    if not raw:
        return set()
    parts = []
    for line in raw.replace("\r", "\n").split("\n"):
        for token in line.split(","):
            val = token.strip()
            if val:
                parts.append(val)
    return set(parts)


def load_preset(preset_root: Path, preset_type: str, preset_file: str) -> Dict:
    ptype = preset_type.strip() or "default"
    fname = preset_file.strip()
    target = preset_root / ptype / fname
    if not target.exists():
        raise FileNotFoundError(f"Preset not found: {target}")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to parse preset: {target} ({exc})")


def list_preset_files(preset_root: Path, preset_type: str) -> List[str]:
    base = preset_root / (preset_type.strip() or "default")
    if not base.exists() or not base.is_dir():
        return []
    return sorted([p.name for p in base.glob("*.json") if p.is_file()])


def parse_tag_file(path: Path) -> TagFile:
    """
    Parse a Danbooru-style tag txt into TagFile object.
    Supports optional '#optional:' and '#warning:' blocks.
    """
    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() or ln.startswith("#warning:")]
    main_line = ""
    optional: List[str] = []
    warning: Optional[str] = None

    for line in lines:
        lower = line.lower()
        if lower.startswith("#optional:"):
            optional = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]
        elif lower.startswith("#warning:"):
            warning = line.split(":", 1)[1].strip()
        elif not main_line:
            main_line = line
        else:
            # Additional non-block lines are appended to main tags
            main_line = f"{main_line}, {line}"

    main_tags = [t.strip() for t in main_line.split(",") if t.strip()] if main_line else []
    return TagFile(path=path, main=main_tags, optional=optional, warning=warning)


def format_tag_file(tag_file: TagFile) -> str:
    lines = []
    if tag_file.main:
        lines.append(", ".join(tag_file.main))
    else:
        lines.append("")
    if tag_file.optional:
        lines.append("#optional: " + ", ".join(tag_file.optional))
    if tag_file.warning:
        lines.append("#warning: " + tag_file.warning)
    return "\n".join(lines).strip() + "\n"


def _dedup(seq: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in seq:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _sort_tags(tags: List[str], priority_groups: List[List[str]]) -> List[str]:
    ordered: List[str] = []
    remaining = list(tags)
    for group in priority_groups or []:
        for tag in group:
            if tag in remaining:
                ordered.append(tag)
                remaining.remove(tag)
    ordered.extend(sorted(remaining))
    return ordered


def _compile_regex(patterns: List[str]) -> List[re.Pattern]:
    out = []
    for pat in patterns or []:
        try:
            out.append(re.compile(pat, flags=re.IGNORECASE))
        except re.error:
            continue
    return out


def _matches_any(tag: str, patterns: List[re.Pattern]) -> bool:
    return any(pat.search(tag) for pat in patterns)


def collect_tag_files(root: Path, recursive: bool, image_exts: List[str], create_missing_txt: bool) -> List[Path]:
    """
    Collect .txt files that have a matching image (by stem). Optionally create
    empty .txt files when missing.
    """
    exts = set([e.lower() if e.startswith(".") else f".{e.lower()}" for e in image_exts or DEFAULT_IMAGE_EXTS])
    tag_paths: Set[Path] = set()
    iterator = root.rglob("*") if recursive else root.iterdir()
    for p in iterator:
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix in exts:
            txt = p.with_suffix(".txt")
            if txt.exists():
                tag_paths.add(txt)
            elif create_missing_txt:
                txt.write_text("", encoding="utf-8")
                tag_paths.add(txt)
        elif suffix == ".txt":
            # Only include .txt that pairs with an image
            has_match = any((p.with_suffix(ext)).exists() for ext in exts)
            if has_match:
                tag_paths.add(p)
    return sorted(tag_paths, key=lambda x: x.as_posix().lower())


def scan_dataset(opts: NormalizeOptions, preset_root: Path, preset_payload: Optional[Dict] = None) -> Dict:
    root = readable_path(str(opts.dataset_path))
    if not root.exists() or not root.is_dir():
        return {"ok": False, "error": f"Dataset folder not found: {root}"}

    try:
        preset = preset_payload or load_preset(preset_root, opts.preset_type, opts.preset_file)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    tag_files = collect_tag_files(root, opts.recursive, opts.image_exts, opts.include_missing_txt)
    parsed: List[TagFile] = []
    counts: Counter = Counter()

    for path in tag_files:
        tf = parse_tag_file(path)
        parsed.append(tf)
        counts.update(tf.main)
        counts.update(tf.optional)

    total_files = len(parsed)
    rules = preset.get("rules", {})
    block_specific = set((rules.get("background_policy") or {}).get("block_specific") or [])
    threshold = opts.background_threshold
    if threshold is None:
        threshold = (rules.get("background_policy") or {}).get("frequency_threshold")

    rare_tags: List[str] = []
    if threshold and total_files:
        for tag, cnt in counts.items():
            if (cnt / total_files) < float(threshold):
                rare_tags.append(tag)

    top_tags = counts.most_common(50)
    sample = [
        {"file": str(tf.path), "main": tf.main, "optional": tf.optional, "warning": tf.warning}
        for tf in parsed[: min(10, len(parsed))]
    ]

    return {
        "ok": True,
        "total_files": total_files,
        "tag_files": len(tag_files),
        "tag_counts": counts,
        "top_tags": top_tags,
        "rare_tags": sorted(rare_tags),
        "block_specific_hits": [tag for tag in block_specific if counts.get(tag)],
        "sample": sample,
    }


def _apply_background_policy(
    main: List[str],
    optional: List[str],
    policy: Dict,
    stats: Dict,
    keep: Set[str],
    move_unknown_to_optional: bool,
    threshold_override: Optional[float],
) -> Tuple[List[str], List[str], List[str]]:
    moved: List[str] = []
    main_out: List[str] = []
    optional_out: List[str] = list(optional)
    if not policy:
        return main, optional, moved

    allow_general = set(policy.get("allow_general") or [])
    block_specific = set(policy.get("block_specific") or [])
    move_specific = bool(policy.get("move_specific_to_optional"))
    threshold = threshold_override if threshold_override is not None else policy.get("frequency_threshold")
    total_files = stats.get("total_files") or 0
    counts: Counter = stats.get("tag_counts") or Counter()

    for tag in main:
        if tag in keep:
            main_out.append(tag)
            continue

        if tag in block_specific:
            if move_specific:
                optional_out.append(tag)
                moved.append(tag)
            # else drop
            continue

        # Frequency-based optional move
        freq = (counts.get(tag, 0) / total_files) if total_files else 1.0
        if threshold and freq < float(threshold) and tag not in allow_general:
            optional_out.append(tag)
            moved.append(tag)
            continue

        # Unknown background tag handling
        if move_unknown_to_optional and tag not in allow_general:
            optional_out.append(tag)
            moved.append(tag)
            continue

        main_out.append(tag)

    return main_out, optional_out, moved


def normalize_record(
    record: TagFile,
    preset: Dict,
    opts: NormalizeOptions,
    stats: Dict,
) -> Tuple[TagFile, Dict]:
    rules = preset.get("rules", {})
    keep_tags = set(rules.get("keep_tags") or [])
    keep_tags.update(opts.extra_keep)
    keep_tags.update(opts.identity_tags)

    remove_tags = set(rules.get("remove_tags") or [])
    remove_tags.update(opts.extra_remove)
    remove_regex = _compile_regex(rules.get("remove_regex") or [])

    replace_map: Dict[str, str] = rules.get("replace_map") or {}
    optional_handling = rules.get("optional_handling") or {}
    move_opt_tags = set(optional_handling.get("move_to_optional_tags") or [])
    move_opt_regex = _compile_regex(optional_handling.get("move_to_optional_regex") or [])

    sort_cfg = rules.get("sort") or {}
    sort_enabled = bool(sort_cfg.get("enabled")) and opts.normalize_order
    priority_groups = sort_cfg.get("priority_groups") or []

    policy = rules.get("background_policy") or {}
    background_enabled = bool(policy.get("enabled"))

    actions: Dict[str, int] = {"removed": 0, "replaced": 0, "dedup": 0, "moved_optional": 0, "sorted": 0}

    before_text = format_tag_file(record)
    main = list(record.main)
    optional = list(record.optional)
    pinned_raw = [t for t in (opts.pinned_tags or []) if t]
    pinned_order = _dedup(pinned_raw)
    pinned_present = [t for t in pinned_order if t in main or t in optional]
    if pinned_present:
        main = [t for t in main if t not in pinned_present]
        optional = [t for t in optional if t not in pinned_present]
        keep_tags.update(pinned_present)

    # 1) trim
    if rules.get("trim"):
        main = [t.strip() for t in main if t.strip()]
        optional = [t.strip() for t in optional if t.strip()]

    # 2) replace map
    if replace_map:
        new_main = [replace_map.get(t, t) for t in main]
        new_optional = [replace_map.get(t, t) for t in optional]
        actions["replaced"] = sum(1 for a, b in zip(main, new_main) if a != b) + sum(
            1 for a, b in zip(optional, new_optional) if a != b
        )
        main, optional = new_main, new_optional

    # 3) remove rules (keep wins)
    filtered_main: List[str] = []
    filtered_optional: List[str] = []
    for tag in main:
        if tag in keep_tags:
            filtered_main.append(tag)
            continue
        if tag in remove_tags or _matches_any(tag, remove_regex):
            actions["removed"] += 1
            continue
        filtered_main.append(tag)
    for tag in optional:
        if tag in keep_tags:
            filtered_optional.append(tag)
            continue
        if tag in remove_tags or _matches_any(tag, remove_regex):
            actions["removed"] += 1
            continue
        filtered_optional.append(tag)
    main = filtered_main
    optional = filtered_optional

    # 4) dedup main/optional
    if rules.get("dedup"):
        dedup_main = _dedup(main)
        dedup_optional = _dedup(optional)
        actions["dedup"] = (len(main) - len(dedup_main)) + (len(optional) - len(dedup_optional))
        main, optional = dedup_main, dedup_optional

    # 5) optional handling specific tags
    moved_now: List[str] = []
    new_main: List[str] = []
    for tag in main:
        if tag in keep_tags:
            new_main.append(tag)
        elif tag in move_opt_tags or _matches_any(tag, move_opt_regex):
            optional.append(tag)
            moved_now.append(tag)
        else:
            new_main.append(tag)
    if moved_now:
        actions["moved_optional"] += len(moved_now)
    main = new_main

    # 5b) background policy
    if background_enabled:
        main, optional, moved_bg = _apply_background_policy(
            main,
            optional,
            policy,
            stats,
            keep_tags,
            opts.move_unknown_background_to_optional,
            opts.background_threshold,
        )
        actions["moved_optional"] += len(moved_bg)

    # 6) sort
    if sort_enabled:
        main = _sort_tags(main, priority_groups)
        optional = _sort_tags(optional, priority_groups)
        actions["sorted"] = 1

    # Final dedup to catch duplicates introduced during moves
    main = _dedup(main)
    optional = _dedup(optional)

    if pinned_present:
        main = _dedup(pinned_present + main)

    after = TagFile(path=record.path, main=main, optional=optional, warning=record.warning)
    after_text = format_tag_file(after)

    changed = before_text.strip() != after_text.strip()
    return after, {"before": before_text, "after": after_text, "actions": actions, "changed": changed}


def dry_run(opts: NormalizeOptions, preset_root: Path, preset_payload: Optional[Dict] = None) -> Dict:
    scan = scan_dataset(opts, preset_root, preset_payload=preset_payload)
    if not scan.get("ok"):
        return scan

    preset = preset_payload or load_preset(preset_root, opts.preset_type, opts.preset_file)
    tag_files = collect_tag_files(
        readable_path(str(opts.dataset_path)), opts.recursive, opts.image_exts, opts.include_missing_txt
    )
    parsed = [parse_tag_file(p) for p in tag_files]
    stats = {
        "total_files": scan.get("total_files", len(parsed)),
        "tag_counts": scan.get("tag_counts") or Counter(),
    }

    previews = []
    for tf in parsed[: opts.preview_limit]:
        after, meta = normalize_record(tf, preset, opts, stats)
        previews.append(
            {
                "file": str(tf.path),
                "before": meta["before"],
                "after": meta["after"],
                "changed": meta["changed"],
                "actions": meta["actions"],
            }
        )

    return {"ok": True, "stats": {"total_files": stats["total_files"], "tag_files": len(tag_files)}, "previews": previews}


def _make_backup(path: Path) -> Optional[Path]:
    try:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        bak = path.with_suffix(path.suffix + ".bak")
        if bak.exists():
            bak = path.with_suffix(path.suffix + f".{stamp}.bak")
        shutil.copy2(path, bak)
        return bak
    except Exception:
        return None


def apply_normalization(opts: NormalizeOptions, preset_root: Path, preset_payload: Optional[Dict] = None) -> Dict:
    scan = scan_dataset(opts, preset_root, preset_payload=preset_payload)
    if not scan.get("ok"):
        return scan

    preset = preset_payload or load_preset(preset_root, opts.preset_type, opts.preset_file)
    tag_files = collect_tag_files(
        readable_path(str(opts.dataset_path)), opts.recursive, opts.image_exts, opts.include_missing_txt
    )
    parsed = [parse_tag_file(p) for p in tag_files]
    stats = {
        "total_files": scan.get("total_files", len(parsed)),
        "tag_counts": scan.get("tag_counts") or Counter(),
    }

    changed_files = 0
    actions_total = Counter()
    backups_made = 0

    for tf in parsed:
        after, meta = normalize_record(tf, preset, opts, stats)
        if not meta["changed"]:
            continue
        if opts.backup_enabled:
            if _make_backup(tf.path):
                backups_made += 1
        tf.path.write_text(meta["after"], encoding="utf-8")
        changed_files += 1
        actions_total.update(meta["actions"])

    summary = {
        "ok": True,
        "total_files": len(parsed),
        "changed_files": changed_files,
        "backups_made": backups_made,
        "actions": dict(actions_total),
    }
    return summary
