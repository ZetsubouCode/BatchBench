"""Caption inspection and guarded manual edits for Dataset Health."""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict


def _caption_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    rel = Path(str(relative or '').replace('\\', '/'))
    if not relative or rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Invalid caption path.')
    if rel.suffix.lower() != '.txt':
        rel = rel.with_suffix('.txt')
    target = (root / rel).resolve()
    if root not in target.parents:
        raise ValueError('Caption path is outside the dataset.')
    return target


def inspect_pair_image(root: Path, relative: str) -> Path | None:
    """Find the paired image within a dataset without allowing path traversal."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset folder not found: {root}')
    rel = Path(str(relative or '').replace('\\', '/'))
    if not relative or rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Invalid pair path.')
    stem = rel.with_suffix('')
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
    for candidate in sorted((root / stem.parent).glob(stem.name + '.*')):
        resolved = candidate.resolve()
        if root in resolved.parents and candidate.suffix.lower() in extensions and resolved.is_file():
            return resolved
    return None


def inspect_caption(root: Path, relative: str) -> Dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f'Dataset folder not found: {root}')
    target = _caption_path(root, relative)
    exists = target.exists()
    if exists and not target.is_file():
        raise ValueError('Caption path is not a file.')
    content = target.read_text(encoding='utf-8-sig') if exists else ''
    old_hash = hashlib.sha256(target.read_bytes()).hexdigest() if exists else None
    return {'ok': True, 'relative_path': target.relative_to(root.resolve()).as_posix(),
            'exists': exists, 'content': content, 'fingerprint': old_hash}


def save_caption(root: Path, relative: str, content: str, expected_hash: str | None) -> Dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f'Dataset folder not found: {root}')
    target = _caption_path(root, relative)
    if target.exists() and not target.is_file():
        raise ValueError('Caption path is not a file.')
    current_hash = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
    if current_hash != expected_hash:
        raise ValueError('Caption changed since opening. Inspect it again before saving.')
    if not target.parent.is_dir():
        raise ValueError('Caption parent folder is missing.')
    normalized = str(content).replace('\r\n', '\n').replace('\r', '\n')
    fd, temp_name = tempfile.mkstemp(prefix='.batchbench-caption-', suffix='.tmp', dir=str(target.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(normalized)
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return inspect_caption(root, relative)
