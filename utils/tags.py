"""Conversions between BatchBench caption tags and Danbooru lookup keys.

These helpers are deliberately for semantic tag values only.  They must not be
used for identifiers, paths, regexes, or user-defined activation tokens unless
the caller has explicitly decided that the value is an ordinary tag.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional, Set


_WHITESPACE_RE = re.compile(r"\s+")
_UNDERSCORE_RUN_RE = re.compile(r"_+")


def to_caption_tag(raw: Any) -> str:
    """Return the canonical human-facing form of one descriptive tag."""
    value = str(raw or "").strip()
    if not value:
        return ""
    value = _UNDERSCORE_RUN_RE.sub(" ", value)
    return _WHITESPACE_RE.sub(" ", value).strip()


def to_danbooru_tag(raw: Any) -> str:
    """Return the lowercase underscore key expected by Danbooru/catalog code."""
    value = to_caption_tag(raw).lower()
    if not value:
        return ""
    return _UNDERSCORE_RUN_RE.sub("_", _WHITESPACE_RE.sub("_", value)).strip("_")


def tag_compare_key(raw: Any) -> str:
    """Representation-independent comparison key for a descriptive tag."""
    return to_danbooru_tag(raw)


def normalize_caption_tags(
    tags: Iterable[Any],
    dedupe: bool = True,
    protected_literals: Optional[Iterable[Any]] = None,
) -> List[str]:
    """Normalize tags for captions while preserving configured literal tokens.

    Protected values are matched exactly after trimming and retain their exact
    underscore spelling.  Dedupe still treats their legacy/spaced counterpart
    as the same logical value.
    """
    protected: Set[str] = {
        str(value).strip() for value in (protected_literals or ()) if str(value).strip()
    }
    out: List[str] = []
    seen: Set[str] = set()
    for raw in tags or ():
        literal = str(raw or "").strip()
        value = literal if literal in protected else to_caption_tag(literal)
        if not value:
            continue
        key = tag_compare_key(value)
        if dedupe and key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out
