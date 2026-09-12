from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class ModelLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class TagMetadata:
    tag_key: str
    model_category: str
    category_id: Optional[int] = None
    recommended_threshold: Optional[float] = None


@dataclass(frozen=True)
class TagPrediction:
    tag_key: str
    score: float
    model_category: str
    recommended_threshold: Optional[float] = None


class TaggerAdapter:
    family = "base"
    preprocessing_label = "base"
    channel_order = "RGB"

    def required_files(self) -> Sequence[str]:
        raise NotImplementedError

    def validate_model_dir(self, model_dir: Path) -> List[str]:
        return [name for name in self.required_files() if not (model_dir / name).is_file()]

    def model_status(self, model_dir: Path) -> Dict[str, Any]:
        missing = self.validate_model_dir(model_dir)
        return {"ready": not missing, "missing": missing, "path": str(model_dir)}

    def load(self, model_dir: Path, *, device: str = "cpu") -> Any:
        raise NotImplementedError

    def preprocess(self, image: Any) -> Any:
        raise NotImplementedError

    def predict(self, loaded: Any, images: Sequence[Any]) -> List[List[TagPrediction]]:
        raise NotImplementedError

    def tag_metadata(self, model_dir: Path) -> List[TagMetadata]:
        raise NotImplementedError

    def optimized_threshold(self, metadata: TagMetadata, fallback: float) -> float:
        return metadata.recommended_threshold if metadata.recommended_threshold is not None else fallback
