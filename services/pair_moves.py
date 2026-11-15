"""Preview and validate Dataset Tag Editor image/sidecar moves."""

import hashlib
import json
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

_PREVIEWS: Dict[str, tuple[float, str]] = {}
_LOCK = threading.Lock()
_MOVE_LOCK = threading.Lock()
_TTL = 15 * 60
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff', '.avif'}


def destination(root: Path, image: Path, target: Path) -> Path:
    """Staging mirrors dataset paths; returning strips only the leading _temp."""
    root, image, target = root.resolve(), image.resolve(), target.resolve()
    rel = image.relative_to(root)
    if target == root / '_temp' and rel.parts[0].lower() != '_temp':
        target = target / rel.parent
    elif target == root and rel.parts[0].lower() == '_temp':
        target = root / Path(*rel.parts[1:]).parent
    if target != root and root not in target.resolve().parents:
        raise ValueError('Destination is outside the dataset.')
    return target


def move(root: Path, src_rel: str, dst_rel: str, preserve_structure: bool = True):
    """Move one image and existing sidecars, rolling back an incomplete pair."""
    with _MOVE_LOCK:
        return _move(root, src_rel, dst_rel, preserve_structure)


def _move(root: Path, src_rel: str, dst_rel: str, preserve_structure: bool):
    root = root.resolve()
    if (root / src_rel).is_symlink():
        return {'ok': False, 'error': 'Symbolic-link images cannot be moved'}, 400
    image = (root / src_rel).resolve()
    target = (root / dst_rel).resolve()
    if root not in image.parents or (target != root and root not in target.parents):
        return {'ok': False, 'error': 'Invalid dataset path'}, 400
    if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
        return {'ok': False, 'error': 'Source image not found or unsupported image extension'}, 404
    moved = []
    try:
        if preserve_structure:
            target = destination(root, image, target)
        if image.parent == target:
            return {'ok': True, 'moved': False, 'rel': src_rel, 'warnings': []}, 200
        target.mkdir(parents=True, exist_ok=True)
        sidecars = _sidecars(image)
        occupied = {path.name.casefold() for path in target.iterdir()}
        bump = 0
        while True:
            stem = image.stem if not bump else f'{image.stem}_{bump}'
            # Reserve the caption name even when the source caption is missing.
            names = [stem + image.suffix, stem + '.txt'] + [stem + suffix for _, suffix in sidecars]
            if not any(name.casefold() in occupied for name in names):
                break
            bump += 1
        plan = [(image, target / (stem + image.suffix))]
        plan.extend((path, target / (stem + suffix)) for path, suffix in sidecars)
        for src, dst in plan:
            shutil.move(str(src), str(dst))
            moved.append((src, dst))
        warnings = [] if image.with_suffix('.txt') in [path for path, _ in sidecars] else [f'Missing sidecar: {image.stem}.txt']
        if bump:
            warnings.append(f'Renamed to avoid conflict: {stem}{image.suffix}')
        return {'ok': True, 'moved': True, 'rel': plan[0][1].relative_to(root).as_posix(), 'warnings': warnings}, 200
    except Exception as exc:
        rollback_errors = []
        for src, dst in reversed(moved):
            try:
                shutil.move(str(dst), str(src))
            except Exception as rollback:
                rollback_errors.append(f'{dst}: {rollback}')
        return {'ok': False, 'error': str(exc) + (' | rollback errors: ' + '; '.join(rollback_errors) if rollback_errors else '')}, 500


def _sidecars(image: Path) -> List[tuple[Path, str]]:
    caption = image.with_suffix('.txt')
    candidates = [(caption, '.txt'), (Path(str(caption) + '.bak'), '.txt.bak'),
                  (image.with_suffix('.bak'), '.bak'), (Path(str(image) + '.bak'), image.suffix + '.bak')]
    for path, _ in candidates:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f'Sidecar must be a regular file: {path.name}')
    return [(path, suffix) for path, suffix in dict.fromkeys(candidates) if path.exists()]


def _fingerprint(root: Path, images: List[Path], target: Path) -> str:
    snapshot: Dict[str, Any] = {'root': str(root.resolve()), 'sources': [str(path.resolve()) for path in images],
                                'target': str(target.resolve()), 'files': []}
    for image in images:
        for path in dict.fromkeys([image, image.with_suffix('.txt'), Path(str(image.with_suffix('.txt')) + '.bak'),
                                   image.with_suffix('.bak'), Path(str(image) + '.bak')]):
            if path.exists():
                stat = path.stat()
                snapshot['files'].append((str(path), stat.st_size, stat.st_mtime_ns))
            else:
                snapshot['files'].append((str(path), None))
    destinations = {destination(root, image, target) for image in images}
    snapshot['target_files'] = [(str(path), path.stat().st_size, path.stat().st_mtime_ns)
                               for folder in sorted(destinations) if folder.is_dir()
                               for path in sorted(folder.iterdir())]
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode('utf-8')).hexdigest()


def preview(root: Path, images: List[Path], target: Path) -> Dict[str, Any]:
    root = root.resolve()
    target = target.resolve()
    if not images or not target.is_dir() or root not in target.parents and target != root:
        raise ValueError('Choose files and a valid destination.')
    occupied_by_folder = {}
    captions = missing = conflicts = noops = 0
    for image in images:
        if image.is_symlink() or not image.is_file() or image.suffix.lower() not in IMAGE_EXTS or root not in image.resolve().parents:
            raise ValueError(f'Source file not found in dataset: {image}')
        folder = destination(root, image, target)
        if image.parent.resolve() == folder:
            noops += 1
            continue
        caption = image.with_suffix('.txt')
        captions += int(caption.is_file())
        missing += int(not caption.is_file())
        if folder not in occupied_by_folder:
            occupied_by_folder[folder] = {path.name.casefold() for path in folder.iterdir()} if folder.is_dir() else set()
        occupied = occupied_by_folder[folder]
        suffixes = set(['.txt'] + [suffix for _, suffix in _sidecars(image)])
        bump = 0
        while True:
            stem = image.stem if bump == 0 else f'{image.stem}_{bump}'
            names = [f'{stem}{image.suffix}'.casefold()] + [f'{stem}{suffix}'.casefold() for suffix in suffixes]
            if not any(name in occupied for name in names):
                occupied.update(names)
                conflicts += int(bump > 0)
                break
            bump += 1
    token = secrets.token_urlsafe(32)
    with _LOCK:
        now = time.monotonic()
        for old, (created, _) in list(_PREVIEWS.items()):
            if now - created > _TTL:
                _PREVIEWS.pop(old, None)
        _PREVIEWS[token] = (now, _fingerprint(root, images, target))
    return {'ok': True, 'preview_token': token, 'images': len(images) - noops,
            'paired_captions': captions, 'missing_captions': missing,
            'conflicts': conflicts, 'noops': noops, 'destination': str(target)}


def consume(root: Path, images: List[Path], target: Path, token: str) -> bool:
    with _LOCK:
        saved = _PREVIEWS.pop(token, None)
    return bool(saved and time.monotonic() - saved[0] <= _TTL and saved[1] == _fingerprint(root, images, target))
