from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple


DEFAULT_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1")


def read_text_best_effort(path: Path, encodings: Iterable[str] = DEFAULT_TEXT_ENCODINGS) -> Tuple[str, str, bool]:
    """
    Read text with fallback encodings.
    Returns: (text, encoding_used, had_decode_replacement)
    """
    data = path.read_bytes()
    for encoding in encodings:
        try:
            return data.decode(encoding), encoding, False
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    # Last-resort decode so callers can continue instead of crashing.
    return data.decode("utf-8", errors="replace"), "utf-8(replace)", True
