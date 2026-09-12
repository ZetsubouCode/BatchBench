from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from services.paths import resource_path, user_path
from utils.tags import tag_compare_key


ALL_CATEGORIES = ("action", "attire", "count", "expression", "feature", "meme", "meta", "object", "other", "setting", "style")
DEFAULT_INCLUDED = frozenset({"action", "expression", "object", "setting", "other"})
DEFAULT_EXCLUDED = frozenset(set(ALL_CATEGORIES) - set(DEFAULT_INCLUDED))
MODEL_BLOCKED_CATEGORIES = frozenset({"character", "rating", "meta", "artist", "copyright"})
_CACHE: Dict[str, Tuple[Tuple[Tuple[str, int, int], ...], "Jio7Classification"]] = {}


def _normalize_key(tag: str) -> str:
    return tag_compare_key(tag).replace(" ", "_")


def default_data_dir() -> Optional[Path]:
    installed = user_path("data", "jio7", "current")
    if installed.is_dir():
        return installed
    bundled = resource_path("data", "jio7", "v1")
    return bundled if bundled.is_dir() else None


@dataclass(frozen=True)
class Jio7Classification:
    source_dir: Path
    categories: Dict[str, str]
    counts: Dict[str, int]
    requiring: Dict[str, Tuple[str, ...]]
    version: str
    missing_categories: Tuple[str, ...]
    source_rows: int = 0

    def category_for(self, tag: str) -> Optional[str]:
        return self.categories.get(_normalize_key(tag))

    def required_attires_for(self, tag: str) -> Tuple[str, ...]:
        return self.requiring.get(_normalize_key(tag), ())

    def is_allowed(
        self,
        tag: str,
        *,
        included: Iterable[str] = DEFAULT_INCLUDED,
        strict: bool = True,
        model_category: str = "general",
        explicit_keep: Iterable[str] = (),
    ) -> bool:
        key = _normalize_key(tag)
        if key in {_normalize_key(item) for item in explicit_keep}:
            return True
        if str(model_category or "unknown").strip().lower() in MODEL_BLOCKED_CATEGORIES:
            return False
        category = self.categories.get(key)
        if category is None:
            return not strict
        return category in {str(item).strip().lower() for item in included}


def _signature(path: Path) -> Tuple[Tuple[str, int, int], ...]:
    rows = []
    for file in sorted(path.glob("*.csv")) + ([path / "requiring.txt"] if (path / "requiring.txt").is_file() else []):
        stat = file.stat()
        rows.append((file.name, stat.st_size, stat.st_mtime_ns))
    return tuple(rows)


def load(source_dir: Optional[Path] = None, *, refresh: bool = False) -> Optional[Jio7Classification]:
    source = Path(source_dir).resolve() if source_dir else default_data_dir()
    if source is None or not source.is_dir():
        return None
    cache_key = str(source)
    signature = _signature(source)
    cached = _CACHE.get(cache_key)
    if cached and not refresh and cached[0] == signature:
        return cached[1]

    categories: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    digest = hashlib.sha256()
    missing: List[str] = []
    source_rows = 0
    for category in ALL_CATEGORIES:
        path = source / f"{category}.csv"
        if not path.is_file():
            missing.append(category)
            continue
        data = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(data)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                tag = _normalize_key(str(row.get("tag") or ""))
                if not tag:
                    continue
                source_rows += 1
                categories.setdefault(tag, category)
                try:
                    counts[tag] = max(counts.get(tag, 0), int(str(row.get("count") or "0")))
                except ValueError:
                    counts.setdefault(tag, 0)

    requiring: Dict[str, Tuple[str, ...]] = {}
    requiring_path = source / "requiring.txt"
    if requiring_path.is_file():
        data = requiring_path.read_bytes()
        digest.update(requiring_path.name.encode("utf-8"))
        digest.update(data)
        for raw in data.decode("utf-8-sig").splitlines():
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            key = _normalize_key(str(row.get("tag") or ""))
            if key:
                requiring[key] = tuple(_normalize_key(item) for item in row.get("required_attires") or [] if _normalize_key(item))

    result = Jio7Classification(source, categories, counts, requiring, digest.hexdigest()[:16], tuple(missing), source_rows)
    _CACHE[cache_key] = (signature, result)
    return result


def status(source_dir: Optional[Path] = None) -> Dict[str, object]:
    classifier = load(source_dir)
    if classifier is None:
        return {"ready": False, "rows": 0, "version": "", "missing_categories": list(ALL_CATEGORIES)}
    category_counts: Dict[str, int] = {}
    for value in classifier.categories.values():
        category_counts[value] = category_counts.get(value, 0) + 1
    return {
        "ready": True,
        "path": str(classifier.source_dir),
        "rows": classifier.source_rows,
        "unique_tags": len(classifier.categories),
        "category_counts": category_counts,
        "requiring_rows": len(classifier.requiring),
        "version": classifier.version,
        "missing_categories": list(classifier.missing_categories),
    }
