"""Preview and validate Dataset Tag Editor image/sidecar moves."""

import hashlib
import json
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

_PREVIEWS: Dict[str, tuple[float, str]] = {}
_LOCK = threading.Lock()
_TTL = 15 * 60


def _sidecars(image: Path) -> List[tuple[Path, str]]:
    caption = image.with_suffix('.txt')
    candidates = [(caption, '.txt'), (Path(str(caption) + '.bak'), '.txt.bak'),
                  (image.with_suffix('.bak'), '.bak'), (Path(str(image) + '.bak'), image.suffix + '.bak')]
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
    if target.is_dir():
        snapshot['target_files'] = [(path.name, path.stat().st_size, path.stat().st_mtime_ns)
                                    for path in sorted(target.iterdir(), key=lambda item: item.name.lower()) if path.is_file()]
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode('utf-8')).hexdigest()


def preview(root: Path, images: List[Path], target: Path) -> Dict[str, Any]:
    root = root.resolve()
    target = target.resolve()
    if not images or not target.is_dir() or root not in target.parents and target != root:
        raise ValueError('Choose files and a valid destination.')
    occupied = {path.name.lower() for path in target.iterdir()}
    captions = missing = conflicts = noops = 0
    for image in images:
        if not image.is_file() or root not in image.resolve().parents:
            raise ValueError(f'Source file not found in dataset: {image}')
        if image.parent.resolve() == target:
            noops += 1
            continue
        caption = image.with_suffix('.txt')
        captions += int(caption.is_file())
        missing += int(not caption.is_file())
        suffixes = [suffix for _, suffix in _sidecars(image)]
        bump = 0
        while True:
            stem = image.stem if bump == 0 else f'{image.stem}_{bump}'
            names = [f'{stem}{image.suffix}'.lower()] + [f'{stem}{suffix}'.lower() for suffix in suffixes]
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
