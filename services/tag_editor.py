from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import re
import shutil
import zipfile

from utils.dataset import join_tags
from utils.io import readable_path
from utils.parse import parse_bool, parse_exts, parse_tag_list
from utils.text_io import read_text_best_effort
from utils.tool_result import build_tool_result

EDIT_MODES = {"insert", "delete", "replace", "dedup"}
DEFAULT_PROJECT_PROMPT = """trigger_word

appearance:
hair_color, eye_color, hairstyle

accessory:
hair_ornament, necklace, earrings

outfit:
top, bottom, footwear

optional:
expression, pose, background
"""
_TXT_TAG_CACHE: "OrderedDict[str, Tuple[Tuple[int, int], List[str]]]" = OrderedDict()
_TXT_TAG_CACHE_MAX = 4096


def _sanitize_tag(t: str) -> str:
    t = (t or "").strip()
    if not t:
        return ""
    return re.sub(r"\s+", "_", t)


def _dedup_tags(tags: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for raw in tags:
        tag = _sanitize_tag(raw)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def _normalize_tags(raw_tags: List[str], dedupe: bool = False) -> List[str]:
    tags = [tag for tag in (_sanitize_tag(x) for x in raw_tags) if tag]
    return _dedup_tags(tags) if dedupe else tags


def _cache_put(key: str, value: Tuple[Tuple[int, int], List[str]]):
    _TXT_TAG_CACHE[key] = value
    _TXT_TAG_CACHE.move_to_end(key)
    while len(_TXT_TAG_CACHE) > _TXT_TAG_CACHE_MAX:
        _TXT_TAG_CACHE.popitem(last=False)


def _read_text_tags(path: Path) -> List[str]:
    try:
        text, _, _ = read_text_best_effort(path)
    except Exception:
        return []
    return _normalize_tags(parse_tag_list(text, dedupe=False), dedupe=False)


def _read_tags_cached(txt_path: Path) -> List[str]:
    key = str(txt_path)
    try:
        st = txt_path.stat()
    except Exception:
        return []
    sig = (st.st_mtime_ns, st.st_size)
    cached = _TXT_TAG_CACHE.get(key)
    if cached and cached[0] == sig:
        _TXT_TAG_CACHE.move_to_end(key)
        return list(cached[1])
    tags = _read_text_tags(txt_path)
    _cache_put(key, (sig, list(tags)))
    return tags


def _invalidate_cache(txt_path: Path):
    _TXT_TAG_CACHE.pop(str(txt_path), None)


def _normalize_exts(exts: List[str]) -> Set[str]:
    out: Set[str] = set()
    for raw in exts or []:
        val = (raw or "").strip().lower()
        if not val:
            continue
        if not val.startswith("."):
            val = "." + val
        out.add(val)
    return out


def _list_images(
    folder_path: Path,
    exts: List[str],
    recursive: bool = False,
    exclude_dir: Optional[Path] = None,
) -> List[Path]:
    if not folder_path.exists() or not folder_path.is_dir():
        return []
    extset = _normalize_exts(exts)
    if not extset:
        return []
    if recursive:
        images: List[Path] = []
        for p in folder_path.rglob("*"):
            if not p.is_file():
                continue
            if exclude_dir and exclude_dir in p.parents:
                continue
            if p.suffix.lower() in extset:
                images.append(p)
    else:
        images = [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in extset]
    return sorted(images, key=lambda p: str(p).lower())


def _next_pair_target(folder: Path, stem: str, image_suffix: str) -> Tuple[Path, Path, bool]:
    bump = 0
    while True:
        suffix = "" if bump == 0 else f"_{bump}"
        candidate_stem = f"{stem}{suffix}"
        dst_img = folder / f"{candidate_stem}{image_suffix}"
        dst_txt = folder / f"{candidate_stem}.txt"
        if not dst_img.exists() and not dst_txt.exists():
            return dst_img, dst_txt, bump > 0
        bump += 1


def _move_pair_transaction(
    src_img: Path,
    src_txt: Path,
    dst_img: Path,
    dst_txt: Path,
    require_txt: bool,
) -> Tuple[bool, str]:
    moved: List[Tuple[Path, Path]] = []
    had_txt = src_txt.exists()
    if require_txt and not had_txt:
        return False, f"Missing paired .txt for {src_img.name}"

    try:
        shutil.move(str(src_img), str(dst_img))
        moved.append((dst_img, src_img))
        if had_txt:
            shutil.move(str(src_txt), str(dst_txt))
            moved.append((dst_txt, src_txt))
        return True, ""
    except Exception as exc:
        rollback_errors: List[str] = []
        for moved_path, original_path in reversed(moved):
            try:
                if not moved_path.exists():
                    continue
                if original_path.exists():
                    rollback_errors.append(
                        f"cannot rollback {moved_path.name} because source already exists: {original_path}"
                    )
                    continue
                shutil.move(str(moved_path), str(original_path))
            except Exception as rollback_exc:
                rollback_errors.append(f"{moved_path.name}: {rollback_exc}")
        message = str(exc)
        if rollback_errors:
            message = f"{message} | rollback issues: {'; '.join(rollback_errors)}"
        return False, message


def _copy_pair_with_txt(src_img: Path, dst_img: Path, dst_txt: Path, txt_content: str) -> Tuple[bool, str]:
    copied: List[Path] = []
    try:
        shutil.copy2(src_img, dst_img)
        copied.append(dst_img)
        dst_txt.write_text(txt_content, encoding="utf-8")
        copied.append(dst_txt)
        return True, ""
    except Exception as exc:
        for path in reversed(copied):
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        return False, str(exc)


def _copy_image_only(src_img: Path, dst_img: Path) -> Tuple[bool, str]:
    try:
        shutil.copy2(src_img, dst_img)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _next_unique_file(path: Path) -> Tuple[Path, bool]:
    if not path.exists():
        return path, False
    bump = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{bump}{path.suffix}")
        if not candidate.exists():
            return candidate, True
        bump += 1


def _created_ts(path: Path) -> float:
    try:
        return float(path.stat().st_ctime)
    except Exception:
        return 0.0


def _is_temp_relative(rel: Path) -> bool:
    parts = rel.parts
    return len(parts) >= 2 and parts[0].lower() == "_temp"


def _serialize_image_item(root_folder: Path, img: Path, tag_limit: int, area: str = "dataset") -> dict:
    txt = img.with_suffix(".txt")
    tags: List[str] = []
    has_txt = False
    if txt.exists():
        has_txt = True
        tags = _read_tags_cached(txt)
    tag_count = len(tags)
    if tag_limit > 0 and len(tags) > tag_limit:
        tags = tags[:tag_limit]
    rel_path = img.relative_to(root_folder)
    rel = rel_path.as_posix()
    parent_rel = rel_path.parent.as_posix() if rel_path.parent != Path(".") else ""
    return {
        "name": img.name,
        "rel": rel,
        "parent_rel": parent_rel,
        "area": area,
        "tags": tags,
        "tag_count": tag_count,
        "tags_truncated": tag_count > len(tags),
        "has_txt": has_txt,
        "in_temp": _is_temp_relative(rel_path),
        "created_ts": _created_ts(img),
    }


def _make_serializable_path_info(paths: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "project_root": str(paths["project_root"]),
        "database_root": str(paths["database_root"]),
        "dataset_root": str(paths["dataset_root"]),
        "temp_root": str(paths["temp_root"]),
        "prompt_path": str(paths["prompt_path"]),
        "normalized_from_legacy": bool(paths["normalized_from_legacy"]),
        "normalized_input": str(paths["normalized_input"]),
    }


def _clean_messages(*items: str) -> List[str]:
    return [str(item).strip() for item in items if str(item or "").strip()]


def _is_image_file(path: Path, extset: Set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in extset


def _image_name_index(images: List[Path]) -> Set[str]:
    return {img.name.lower() for img in images if img and img.is_file()}


def resolve_project_paths(raw_folder: Path) -> Dict[str, Any]:
    normalized_input = readable_path(str(raw_folder or "")).expanduser()
    project_root = normalized_input
    normalized_from_legacy = False
    if project_root.name.lower() in {"dataset", "database"} and project_root.parent != project_root:
        project_root = project_root.parent
        normalized_from_legacy = True
    return {
        "project_root": project_root,
        "database_root": project_root / "database",
        "dataset_root": project_root / "dataset",
        "temp_root": project_root / "dataset" / "_temp",
        "prompt_path": project_root / "prompt.txt",
        "normalized_from_legacy": normalized_from_legacy,
        "normalized_input": str(normalized_input),
    }


def extract_trigger_word(prompt_path: Path) -> str:
    if not prompt_path.exists() or not prompt_path.is_file():
        return ""
    try:
        text, _, _ = read_text_best_effort(prompt_path)
    except Exception:
        return ""
    lines = [line.strip() for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    non_empty = [line for line in lines if line]
    if not non_empty:
        return ""
    trigger_keys = {"trigger", "trigger_word", "trigger word", "activation", "activation_word"}
    for line in non_empty:
        if ":" not in line:
            continue
        left, right = line.split(":", 1)
        if left.strip().lower() in trigger_keys:
            return right.strip().strip(",")
    first = non_empty[0]
    if ":" not in first:
        return first.strip().strip(",")
    left, right = first.split(":", 1)
    if left.strip().lower() in trigger_keys:
        return right.strip().strip(",")
    return ""


def inspect_project_layout(project_root: Path, exts: List[str]) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    root = paths["project_root"]
    extset = _normalize_exts(exts)
    if not str(root).strip():
        return {"ok": False, "error": "Project root is required", **_make_serializable_path_info(paths)}
    if root.exists() and not root.is_dir():
        return {"ok": False, "error": f"Project root is not a folder: {root}", **_make_serializable_path_info(paths)}

    database_exists = paths["database_root"].is_dir()
    dataset_exists = paths["dataset_root"].is_dir()
    temp_exists = paths["temp_root"].is_dir()
    prompt_exists = paths["prompt_path"].exists()
    missing = []
    if not database_exists:
        missing.append("database")
    if not dataset_exists:
        missing.append("dataset")
    if not temp_exists:
        missing.append("dataset/_temp")
    if not prompt_exists:
        missing.append("prompt.txt")

    root_images: List[Path] = []
    root_non_images = 0
    if root.exists() and root.is_dir() and extset:
        for path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if _is_image_file(path, extset):
                root_images.append(path)
            elif path.is_file():
                root_non_images += 1

    database_images = _list_images(paths["database_root"], exts, recursive=True)
    dataset_images = _list_images(paths["dataset_root"], exts, recursive=True, exclude_dir=paths["temp_root"])
    database_names = _image_name_index(database_images)
    dataset_names = _image_name_index(dataset_images)
    missing_root_in_database = [img.name for img in root_images if img.name.lower() not in database_names]
    missing_root_in_dataset = [img.name for img in root_images if img.name.lower() not in dataset_names]
    dataset_missing_txt_files = [
        img.relative_to(paths["dataset_root"]).with_suffix(".txt").as_posix()
        for img in dataset_images
        if not img.with_suffix(".txt").exists()
    ]
    dataset_missing_txt = len(dataset_missing_txt_files)
    dataset_existing_txt = max(0, len(dataset_images) - dataset_missing_txt)

    if missing_root_in_database:
        missing.append(f"database(root images:{len(missing_root_in_database)})")
    if missing_root_in_dataset:
        missing.append(f"dataset(root images:{len(missing_root_in_dataset)})")
    if dataset_missing_txt:
        missing.append(f"dataset(txt pairs:{dataset_missing_txt})")

    trigger_word = extract_trigger_word(paths["prompt_path"])
    preview_examples: List[Dict[str, str]] = []
    conflict_count = 0
    missing_database_name_index = {name.lower() for name in missing_root_in_database}
    for img in root_images:
        if img.name.lower() not in missing_database_name_index:
            continue
        if paths["database_root"].exists():
            dst_img = paths["database_root"] / img.name
            if paths["database_root"].joinpath(img.name).exists():
                conflict_count += 1
            preview_examples.append({"source": img.name, "image": dst_img.relative_to(root).as_posix()})
        else:
            preview_examples.append({"source": img.name, "image": f"database/{img.name}"})

    ready = root.exists() and root.is_dir() and not missing
    warnings: List[str] = []
    if root.exists() and root.is_dir() and not extset:
        warnings.append("No valid image extensions supplied; root-level image scan skipped.")
    if root.exists() and root.is_dir() and not root_images and not ready:
        warnings.append("No root-level source images detected for initialization preview.")
    if dataset_missing_txt:
        warnings.append(f"Dataset has {dataset_missing_txt} image(s) without paired .txt.")
    return {
        "ok": True,
        **_make_serializable_path_info(paths),
        "exists": root.exists(),
        "is_dir": root.is_dir() if root.exists() else False,
        "ready": ready,
        "needs_init": not ready,
        "status": "ready" if ready else ("partial" if root.exists() else "missing"),
        "missing": missing,
        "database_exists": database_exists,
        "dataset_exists": dataset_exists,
        "temp_exists": temp_exists,
        "prompt_exists": prompt_exists,
        "root_image_count": len(root_images),
        "root_non_image_count": root_non_images,
        "database_image_count": len(database_images),
        "dataset_image_count": len(dataset_images),
        "root_images": [img.name for img in root_images[:20]],
        "missing_root_in_database": missing_root_in_database[:50],
        "missing_root_in_dataset": missing_root_in_dataset[:50],
        "missing_dataset_txt": dataset_missing_txt_files[:100],
        "trigger_word": trigger_word,
        "trigger_detected": bool(trigger_word),
        "init_preview": {
            "create_dirs": [part for part in ["database", "dataset", "dataset/_temp"] if part in missing],
            "create_prompt": not prompt_exists,
            "root_images_found": len(root_images),
            "database_images_found": len(database_images),
            "dataset_images_found": len(dataset_images),
            "copy_images": len(missing_root_in_database),
            "copy_dataset_images": len(missing_root_in_dataset),
            "generate_txt": dataset_missing_txt + len(missing_root_in_dataset),
            "existing_txt": dataset_existing_txt,
            "skip_images": 0,
            "conflict_rename": conflict_count,
            "examples": preview_examples[:5],
        },
        "warnings": warnings,
        "info": _clean_messages(
            f"Normalized legacy input to project root: {root}" if paths["normalized_from_legacy"] else "",
            f"Trigger word detected: {trigger_word}" if trigger_word else "",
            (
                f"Missing root images -> database: {len(missing_root_in_database)}, "
                f"dataset: {len(missing_root_in_dataset)}"
                if root_images
                else ""
            ),
        ),
    }


def initialize_project_layout(project_root: Path, exts: List[str], create_prompt: bool = True) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    root = paths["project_root"]
    inspect = inspect_project_layout(root, exts)
    if not inspect.get("ok"):
        return inspect
    if root.exists() and not root.is_dir():
        return {"ok": False, "error": f"Project root is not a folder: {root}", **_make_serializable_path_info(paths)}
    if not root.exists():
        return {"ok": False, "error": f"Project root not found: {root}", **_make_serializable_path_info(paths)}

    logs: List[str] = []
    errors: List[str] = []
    created_dirs: List[str] = []
    copied: List[Dict[str, str]] = []
    copied_dataset: List[Dict[str, str]] = []
    generated_txt: List[str] = []
    skipped: List[str] = []
    renamed: List[Dict[str, str]] = []

    for name, path in (("database", paths["database_root"]), ("dataset", paths["dataset_root"]), ("dataset/_temp", paths["temp_root"])):
        if path.exists():
            if not path.is_dir():
                errors.append(f"{name} exists but is not a folder: {path}")
            else:
                skipped.append(f"{name}: already exists")
            continue
        try:
            path.mkdir(parents=True, exist_ok=False)
            created_dirs.append(name)
            logs.append(f"[mkdir] {name}")
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if create_prompt:
        if paths["prompt_path"].exists():
            skipped.append("prompt.txt: already exists")
        else:
            try:
                paths["prompt_path"].write_text(DEFAULT_PROJECT_PROMPT, encoding="utf-8")
                logs.append("[create] prompt.txt")
            except Exception as exc:
                errors.append(f"prompt.txt: {exc}")

    trigger_word = extract_trigger_word(paths["prompt_path"])
    extset = _normalize_exts(exts)
    root_images = [path for path in sorted(root.iterdir(), key=lambda p: p.name.lower()) if _is_image_file(path, extset)]
    txt_content = trigger_word.strip()

    database_images_before = _list_images(paths["database_root"], exts, recursive=True)
    database_name_index = _image_name_index(database_images_before)
    dataset_images_before = _list_images(paths["dataset_root"], exts, recursive=True, exclude_dir=paths["temp_root"])
    dataset_name_index = _image_name_index(dataset_images_before)

    # Initialization ensures root-level source images exist in database/.
    for src_img in root_images:
        src_key = src_img.name.lower()
        if src_key in database_name_index:
            skipped.append(f"database/{src_img.name}: already exists")
            logs.append(f"[skip] database/{src_img.name} already exists")
            continue
        dst_img = paths["database_root"] / src_img.name
        renamed_flag = False
        if dst_img.exists():
            dst_img, renamed_flag = _next_unique_file(dst_img)
        if renamed_flag:
            renamed.append({"source": src_img.name, "target": dst_img.name, "scope": "database"})
            logs.append(f"[rename][database] {src_img.name} -> {dst_img.name}")
        ok, error = _copy_image_only(src_img, dst_img)
        if not ok:
            errors.append(f"{src_img.name}: {error}")
            logs.append(f"[error] copy {src_img.name}: {error}")
            continue
        database_name_index.add(dst_img.name.lower())
        copied.append({"source": src_img.name, "image": dst_img.name})
        logs.append(f"[copy] {src_img.name} -> database/{dst_img.name}")

    # Initialization also ensures root-level source images exist in dataset/.
    for src_img in root_images:
        src_key = src_img.name.lower()
        if src_key in dataset_name_index:
            skipped.append(f"dataset/{src_img.name}: image already exists")
            logs.append(f"[skip] dataset/{src_img.name} image already exists")
            continue
        dst_img = paths["dataset_root"] / src_img.name
        renamed_flag = False
        if dst_img.exists():
            dst_img, renamed_flag = _next_unique_file(dst_img)
        if renamed_flag:
            renamed.append({"source": src_img.name, "target": dst_img.name, "scope": "dataset"})
            logs.append(f"[rename][dataset] {src_img.name} -> {dst_img.name}")
        ok, error = _copy_image_only(src_img, dst_img)
        if not ok:
            errors.append(f"dataset/{src_img.name}: {error}")
            logs.append(f"[error] copy dataset/{src_img.name}: {error}")
            continue
        dataset_name_index.add(dst_img.name.lower())
        rel_img = dst_img.relative_to(paths["dataset_root"]).as_posix()
        copied_dataset.append({"source": src_img.name, "image": rel_img})
        logs.append(f"[copy] {src_img.name} -> dataset/{rel_img}")

    # Captions are generated for dataset images (excluding _temp).
    dataset_images = _list_images(paths["dataset_root"], exts, recursive=True, exclude_dir=paths["temp_root"])
    for img in dataset_images:
        txt_path = img.with_suffix(".txt")
        rel_txt = txt_path.relative_to(paths["dataset_root"]).as_posix()
        if txt_path.exists():
            skipped.append(f"dataset/{rel_txt}: already exists")
            logs.append(f"[skip] dataset/{rel_txt} already exists")
            continue
        try:
            txt_path.write_text(txt_content, encoding="utf-8")
            generated_txt.append(rel_txt)
            logs.append(f"[txt] dataset/{rel_txt}" + (f" (trigger: {trigger_word})" if trigger_word else " (empty)"))
        except Exception as exc:
            errors.append(f"dataset/{rel_txt}: {exc}")
            logs.append(f"[error] txt dataset/{rel_txt}: {exc}")

    return {
        "ok": len(errors) == 0,
        **_make_serializable_path_info(paths),
        "trigger_word": trigger_word,
        "created_dirs": created_dirs,
        "copied": copied,
        "copied_dataset": copied_dataset,
        "generated_txt": generated_txt,
        "skipped": skipped,
        "renamed": renamed,
        "warnings": _clean_messages(
            f"Trigger word detected: {trigger_word}" if trigger_word else "",
            "Some files were skipped or renamed during initialization." if skipped or renamed else "",
        ),
        "errors": errors,
        "logs": logs,
        "summary": {
            "created_dirs": len(created_dirs),
            "copied_images": len(copied) + len(copied_dataset),
            "copied_database_images": len(copied),
            "copied_dataset_images": len(copied_dataset),
            "generated_txt": len(generated_txt),
            "skipped": len(skipped),
            "renamed": len(renamed),
            "errors": len(errors),
        },
    }


def export_dataset_zip(project_root: Path) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    dataset_root = paths["dataset_root"]
    if not dataset_root.exists() or not dataset_root.is_dir():
        return {"ok": False, "error": f"Dataset folder not found: {dataset_root}", **_make_serializable_path_info(paths)}

    parent_dir = dataset_root.parent
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    zip_path, renamed = _next_unique_file(parent_dir / f"{dataset_root.name}__{stamp}.zip")
    zipped = 0
    excluded_temp = 0
    logs: List[str] = [
        f"[zip] source dataset: {dataset_root}",
        f"[zip] output zip: {zip_path}",
        "[zip] exclude rule: dataset/_temp/**",
    ]

    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(dataset_root.rglob("*"), key=lambda p: str(p).lower()):
                if not path.is_file():
                    continue
                rel = path.relative_to(dataset_root)
                rel_parts = [part.lower() for part in rel.parts]
                if rel_parts and rel_parts[0] == "_temp":
                    excluded_temp += 1
                    continue
                zf.write(path, rel.as_posix())
                zipped += 1
    except Exception as exc:
        try:
            if zip_path.exists():
                zip_path.unlink()
        except Exception:
            pass
        return {"ok": False, "error": f"Dataset zip failed: {exc}", **_make_serializable_path_info(paths)}

    if renamed:
        logs.append(f"[zip] output renamed to avoid collision: {zip_path.name}")
    logs.append(f"[zip] files added: {zipped}")
    logs.append(f"[zip] files excluded from _temp: {excluded_temp}")
    return {
        "ok": True,
        **_make_serializable_path_info(paths),
        "zip_path": str(zip_path),
        "zip_name": zip_path.name,
        "files_zipped": zipped,
        "excluded_temp_files": excluded_temp,
        "logs": logs,
    }


def list_database_images(project_root: Path, exts: List[str], recursive: bool, limit: int, tag_limit: int) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    result = list_images_with_tags(paths["database_root"], exts, recursive=recursive, limit=limit, tag_limit=tag_limit)
    if result.get("ok"):
        result.update({"area": "database", **_make_serializable_path_info(paths)})
    return result


def list_dataset_images(
    project_root: Path,
    exts: List[str],
    recursive: bool,
    limit: int,
    tag_limit: int,
    include_temp: bool = True,
) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    folder = paths["dataset_root"]
    if not folder.exists() or not folder.is_dir():
        return {"ok": False, "error": f"Folder not found: {folder}", **_make_serializable_path_info(paths)}
    images = _list_images(folder, exts, recursive=recursive, exclude_dir=None if include_temp else paths["temp_root"])
    total = len(images)
    if limit and limit > 0:
        images = images[:limit]
    items = [_serialize_image_item(folder, img, tag_limit, area="dataset") for img in images]
    return {"ok": True, "folder": str(folder), "total": total, "images": items, "area": "dataset", **_make_serializable_path_info(paths)}


def list_temp_images(project_root: Path, exts: List[str], recursive: bool, limit: int, tag_limit: int) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    folder = paths["temp_root"]
    if not folder.exists() or not folder.is_dir():
        return {"ok": False, "error": f"Folder not found: {folder}", **_make_serializable_path_info(paths)}
    images = _list_images(folder, exts, recursive=recursive, exclude_dir=None)
    total = len(images)
    if limit and limit > 0:
        images = images[:limit]
    items = [_serialize_image_item(folder, img, tag_limit, area="temp") for img in images]
    return {"ok": True, "folder": str(folder), "total": total, "images": items, "area": "temp", **_make_serializable_path_info(paths)}


def move_database_files_to_temp(project_root: Path, srcs: List[str]) -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    database_root = paths["database_root"]
    temp_root = paths["temp_root"]
    if not database_root.exists() or not database_root.is_dir():
        return {"ok": False, "error": f"Database folder not found: {database_root}", **_make_serializable_path_info(paths)}
    temp_root.mkdir(parents=True, exist_ok=True)
    moved = []
    warnings = []
    errors = []
    seen: Set[str] = set()

    for raw_rel in srcs or []:
        rel = str(raw_rel or "").replace("\\", "/").strip().lstrip("/")
        if not rel or rel in seen:
            continue
        seen.add(rel)
        src_img = database_root / Path(rel)
        try:
            src_img_res = src_img.resolve()
            db_res = database_root.resolve()
        except Exception:
            errors.append({"src": rel, "error": "Path resolution failed"})
            continue
        if db_res not in src_img_res.parents and src_img_res != db_res:
            errors.append({"src": rel, "error": "Invalid relative path"})
            continue
        if not src_img.exists() or not src_img.is_file():
            errors.append({"src": rel, "error": "Source image not found"})
            continue
        src_txt = src_img.with_suffix(".txt")
        rel_parent = Path(rel).parent
        dst_dir = temp_root if rel_parent == Path(".") else temp_root / rel_parent
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            errors.append({"src": rel, "error": str(exc)})
            continue
        dst_img, dst_txt, renamed_flag = _next_pair_target(dst_dir, src_img.stem, src_img.suffix.lower())
        ok, error = _move_pair_transaction(src_img, src_txt, dst_img, dst_txt, require_txt=False)
        if not ok:
            errors.append({"src": rel, "error": error})
            continue
        item = {"src": rel, "rel": dst_img.relative_to(paths["dataset_root"]).as_posix(), "moved": True, "warnings": []}
        if renamed_flag:
            item["warnings"].append(f"Renamed on move: {src_img.name} -> {dst_img.name}")
        if not dst_txt.exists():
            item["warnings"].append(f"Missing sidecar: {src_img.stem}.txt")
        if item["warnings"]:
            warnings.extend(item["warnings"])
        moved.append(item)
    return {"ok": len(errors) == 0, **_make_serializable_path_info(paths), "moved": moved, "warnings": warnings, "errors": errors}


def resolve_image_path(project_root: Path, rel: str, area: str = "database") -> Dict[str, Any]:
    paths = resolve_project_paths(project_root)
    area_key = (area or "database").strip().lower()
    if area_key == "database":
        root = paths["database_root"]
    elif area_key == "dataset":
        root = paths["dataset_root"]
    elif area_key == "temp":
        root = paths["temp_root"]
    else:
        return {"ok": False, "error": f"Unknown area: {area}", **_make_serializable_path_info(paths)}
    rel_path = Path(str(rel or "").replace("\\", "/").lstrip("/"))
    if not str(rel_path) or any(part == ".." for part in rel_path.parts):
        return {"ok": False, "error": "Invalid path", **_make_serializable_path_info(paths)}
    target = root / rel_path
    try:
        target_res = target.resolve()
        root_res = root.resolve()
    except Exception:
        return {"ok": False, "error": "Path resolution failed", **_make_serializable_path_info(paths)}
    if target_res != root_res and root_res not in target_res.parents:
        return {"ok": False, "error": "Invalid path", **_make_serializable_path_info(paths)}
    return {"ok": True, "root": root, "target": target, "area": area_key, **_make_serializable_path_info(paths)}


def scan_tags(folder: Path, exts: List[str], recursive: bool = False) -> dict:
    if not folder or not folder.exists() or not folder.is_dir():
        return {"ok": False, "error": f"Folder not found: {folder}"}
    images = _list_images(folder, exts, recursive=recursive)
    counts: Counter = Counter()
    tag_files = 0
    for img in images:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            continue
        tags = _read_tags_cached(txt)
        tag_files += 1
        if tags:
            counts.update(set(tags))
    items = [{"tag": tag, "count": int(cnt)} for tag, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    return {"ok": True, "folder": str(folder), "total_images": len(images), "tag_files": tag_files, "tags": items}


def list_images_with_tags(folder: Path, exts: List[str], recursive: bool = False, limit: int = 80, tag_limit: int = 200) -> dict:
    if not folder or not folder.exists() or not folder.is_dir():
        return {"ok": False, "error": f"Folder not found: {folder}"}
    images = _list_images(folder, exts, recursive=recursive, exclude_dir=None)
    total = len(images)
    if limit and limit > 0:
        images = images[:limit]
    items = [_serialize_image_item(folder, img, tag_limit) for img in images]
    return {"ok": True, "folder": str(folder), "total": total, "images": items}


def add_tags(txt_path: Path, raw_tags: List[str], backup: bool = True, create_missing_txt: bool = False) -> dict:
    tags_to_add = _dedup_tags(raw_tags or [])
    if not tags_to_add:
        return {"ok": False, "error": "No tags provided"}
    had_txt = txt_path.exists()
    src = ""
    current_tags: List[str] = []
    if had_txt:
        try:
            src, _, _ = read_text_best_effort(txt_path)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        current_tags = _normalize_tags(parse_tag_list(src, dedupe=False), dedupe=False)
    elif not create_missing_txt:
        return {"ok": False, "error": "Missing .txt"}
    next_tags = list(current_tags)
    added: List[str] = []
    for tag in tags_to_add:
        if tag in next_tags:
            continue
        next_tags.append(tag)
        added.append(tag)
    if not had_txt and not next_tags:
        return {"ok": False, "error": "No tags to write"}
    content = join_tags(next_tags)
    changed = (not had_txt) or (content.strip() != src.strip())
    if changed:
        if backup and had_txt:
            try:
                txt_path.with_suffix(txt_path.suffix + ".bak").write_text(src, encoding="utf-8")
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
        try:
            txt_path.write_text(content, encoding="utf-8")
            _invalidate_cache(txt_path)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
    return {"ok": True, "created": not had_txt, "changed": changed, "added": added, "tags": next_tags}


def remove_tag(txt_path: Path, tag: str, backup: bool = True) -> dict:
    if not txt_path or not txt_path.exists() or not txt_path.is_file():
        return {"ok": False, "error": "Missing .txt"}
    try:
        src, _, _ = read_text_best_effort(txt_path)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    tag = _sanitize_tag(tag)
    taglist = _normalize_tags(parse_tag_list(src, dedupe=False), dedupe=False)
    newtags = [t for t in taglist if t != tag]
    removed = len(newtags) != len(taglist)
    if not removed:
        return {"ok": True, "removed": False, "tags": taglist}
    if backup:
        try:
            txt_path.with_suffix(txt_path.suffix + ".bak").write_text(src, encoding="utf-8")
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
    try:
        txt_path.write_text(join_tags(newtags), encoding="utf-8")
        _invalidate_cache(txt_path)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "removed": True, "tags": newtags}


def handle(form, ctx):
    active_tab = "tags"
    raw_folder_text = (form.get("folder", "") or "").strip()
    mode = (form.get("mode", "insert") or "insert").strip().lower()
    tags_field = form.get("tags", "")
    exts = parse_exts(form.get("exts"), default=[".jpg", ".jpeg", ".png", ".webp"])
    backup = parse_bool(form.get("backup"), default=False)
    create_missing_txt = parse_bool(form.get("create_missing_txt"), default=False)
    lines: List[str] = []

    def _done(ok: bool, error: str = ""):
        return build_tool_result(active_tab, lines, ok=ok, error=error)

    if not raw_folder_text:
        lines.append("Project root not provided.")
        return _done(False, "Project root not provided.")
    if mode not in EDIT_MODES and mode != "undo":
        lines.append(f"Unknown mode: {mode}")
        return _done(False, f"Unknown mode: {mode}")

    raw_folder = readable_path(raw_folder_text)
    paths = resolve_project_paths(raw_folder)
    dataset_root = paths["dataset_root"]
    temp_folder = paths["temp_root"]
    compatibility_mode = False
    if (not dataset_root.exists() or not dataset_root.is_dir()) and raw_folder.exists() and raw_folder.is_dir():
        direct_dataset = None
        direct_temp = None
        if raw_folder.name.lower() == "_temp" and raw_folder.parent.exists() and raw_folder.parent.is_dir():
            direct_dataset = raw_folder.parent
            direct_temp = raw_folder
        elif (raw_folder / "_temp").exists() and (raw_folder / "_temp").is_dir():
            direct_dataset = raw_folder
            direct_temp = raw_folder / "_temp"
        if direct_dataset and direct_temp:
            dataset_root = direct_dataset
            temp_folder = direct_temp
            compatibility_mode = True
            lines.append(f"Compatibility mode: treating {raw_folder} as dataset root.")
    if not dataset_root.exists() or not dataset_root.is_dir():
        lines.append(f"Dataset folder not found: {dataset_root}")
        return _done(False, f"Dataset folder not found: {dataset_root}")
    if not temp_folder.exists() or not temp_folder.is_dir():
        lines.append(f"Temp folder not found: {temp_folder}")
        return _done(False, f"Temp folder not found: {temp_folder}")
    if (not compatibility_mode) and (not paths.get("normalized_from_legacy")):
        project_state = inspect_project_layout(paths["project_root"], exts)
        if not project_state.get("ok"):
            error = project_state.get("error") or "Project setup validation failed."
            lines.append(error)
            return _done(False, error)
        if not project_state.get("ready"):
            missing_items = project_state.get("missing") or []
            missing_msg = ", ".join(missing_items) if missing_items else "project setup incomplete"
            lines.append(f"Project initialization required: {missing_msg}")
            return _done(False, "Project initialization required. Use Initialize Project first.")

    def _parse_tag_input(raw: str) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for line in (raw or "").replace("\r", "\n").split("\n"):
            for token in line.split(","):
                tag = _sanitize_tag(token)
                if not tag or tag in seen:
                    continue
                out.append(tag)
                seen.add(tag)
        return out

    def _parse_replace_mapping(raw: str) -> dict:
        mapping = {}
        parts = [p.strip() for p in (raw or "").split(";") if p.strip()]
        for p in parts:
            if "->" not in p:
                continue
            old, new = [x.strip() for x in p.split("->", 1)]
            old = _sanitize_tag(old)
            new = _sanitize_tag(new)
            if old:
                mapping[old] = new
        return mapping

    if mode == "undo":
        images_in_temp = _list_images(temp_folder, exts, recursive=True)
        if not images_in_temp:
            lines.append(f"No image files with {exts} found in temp folder: {temp_folder}")
            return _done(False, f"No image files found in temp folder: {temp_folder}")
        restored = 0
        errors = 0
        for img in sorted(images_in_temp, key=lambda p: p.name.lower()):
            txt = img.with_suffix(".txt")
            had_txt = txt.exists()
            rel_parent = img.parent.relative_to(temp_folder)
            dest_parent = dataset_root / rel_parent
            dest_parent.mkdir(parents=True, exist_ok=True)
            dest_img, dest_txt, _ = _next_pair_target(dest_parent, img.stem, img.suffix.lower())
            ok, error = _move_pair_transaction(img, txt, dest_img, dest_txt, require_txt=False)
            if not ok:
                lines.append(f"[ERROR] restoring {img.name}: {error}")
                errors += 1
                continue
            restored += 1
            lines.append(f"Restored: {dest_img.relative_to(dataset_root)}{' (+ .txt)' if had_txt else ''}")
        lines.append(f"Done. {restored} file(s) restored from {temp_folder} to {dataset_root}. Errors: {errors}.")
        return _done(errors == 0, "" if errors == 0 else f"{errors} restore operation(s) failed.")

    add: List[str] = []
    deltags: Set[str] = set()
    mapping = {}
    if mode == "insert":
        add = _parse_tag_input(tags_field)
    elif mode == "delete":
        deltags = set(_parse_tag_input(tags_field))
    elif mode == "replace":
        mapping = _parse_replace_mapping(tags_field)

    if mode == "insert" and not add:
        lines.append("Insert skipped: no tags provided.")
        return _done(False, "Insert skipped: no tags provided.")
    if mode == "delete" and not deltags:
        lines.append("Delete skipped: no tags provided.")
        return _done(False, "Delete skipped: no tags provided.")
    if mode == "replace" and not mapping:
        lines.append("Replace skipped: no mappings provided.")
        return _done(False, "Replace skipped: no mappings provided.")

    images = _list_images(temp_folder, exts, recursive=True, exclude_dir=None)
    if not images:
        lines.append(f"No image files with {exts} found in: {temp_folder}")
        return _done(False, f"No image files found in: {temp_folder}")

    processed = 0
    errors = 0
    missing_txt_skipped = 0
    missing_txt_created = 0
    for img in images:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            if mode == "insert" and create_missing_txt:
                try:
                    txt.write_text(join_tags(add), encoding="utf-8")
                    _invalidate_cache(txt)
                    lines.append(f"{img.name}: created missing .txt with insert tags")
                    processed += 1
                    missing_txt_created += 1
                except Exception as exc:
                    lines.append(f"[ERROR] creating {txt.name}: {exc}")
                    errors += 1
            else:
                missing_txt_skipped += 1
            continue
        try:
            src, _, _ = read_text_best_effort(txt)
        except Exception as exc:
            lines.append(f"[ERROR] reading {txt.name}: {exc}")
            errors += 1
            continue
        taglist = _normalize_tags(parse_tag_list(src, dedupe=False), dedupe=False)
        newtags = list(taglist)
        action_desc = ""
        if mode == "insert":
            for t in add:
                if t not in newtags:
                    newtags.append(t)
            action_desc = f"insert -> {add}"
        elif mode == "delete":
            newtags = [t for t in taglist if t not in deltags]
            action_desc = f"delete -> {sorted(deltags)}"
        elif mode == "replace":
            newtags = [mapping.get(t, t) for t in taglist]
            action_desc = f"replace -> {mapping}"
        elif mode == "dedup":
            newtags = _dedup_tags(taglist)
            action_desc = f"dedup -> {len(taglist) - len(newtags)} removed"
        content = join_tags(newtags)
        if content.strip() == src.strip():
            continue
        try:
            if backup:
                txt.with_suffix(txt.suffix + ".bak").write_text(src, encoding="utf-8")
            txt.write_text(content, encoding="utf-8")
            _invalidate_cache(txt)
            lines.append(f"{img.name}: {action_desc}")
        except Exception as exc:
            lines.append(f"[ERROR] writing {txt.name}: {exc}")
            errors += 1
            continue
        processed += 1

    if mode == "insert" and missing_txt_skipped:
        lines.append(f"Skipped {missing_txt_skipped} image(s) without paired .txt in {temp_folder}.")
    if mode == "insert" and missing_txt_created:
        lines.append(f"Created {missing_txt_created} new .txt file(s) for images that had no paired tags.")
    lines.append(f"Done. {processed} file(s) updated in {temp_folder}. Errors: {errors}.")
    return _done(errors == 0, "" if errors == 0 else f"{errors} file update(s) failed.")
