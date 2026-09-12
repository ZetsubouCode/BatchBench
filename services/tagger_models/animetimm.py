from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image, ImageOps

from .base import ModelLoadError, TagMetadata, TagPrediction, TaggerAdapter


class AnimeTimmDbv4Adapter(TaggerAdapter):
    family = "animetimm_dbv4"
    preprocessing_label = "animetimm_dbv4 (RGB)"
    channel_order = "RGB"

    def __init__(self) -> None:
        self.input_size = 384
        self.pad_size = 512
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def required_files(self) -> Sequence[str]:
        return ("model.onnx", "selected_tags.csv", "preprocess.json", "categories.json")

    def _load_preprocess(self, model_dir: Path) -> None:
        try:
            payload = json.loads((model_dir / "preprocess.json").read_text(encoding="utf-8"))
            steps = payload.get("test") or payload.get("val") or []
            for step in steps:
                kind = step.get("type")
                if kind == "pad_to_size":
                    size = step.get("size") or [512, 512]
                    self.pad_size = int(size[0])
                elif kind == "center_crop":
                    size = step.get("size") or [384, 384]
                    self.input_size = int(size[0])
                elif kind == "normalize":
                    self.mean = np.asarray(step.get("mean"), dtype=np.float32)
                    self.std = np.asarray(step.get("std"), dtype=np.float32)
        except Exception as exc:
            raise ModelLoadError(f"Invalid CAFormer preprocess.json: {exc}") from exc

    def tag_metadata(self, model_dir: Path) -> List[TagMetadata]:
        out: List[TagMetadata] = []
        with (model_dir / "selected_tags.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                name = str(row.get("name") or row.get("tag") or "").strip()
                if not name:
                    continue
                try:
                    category_id = int(str(row.get("category") or "").strip())
                except ValueError:
                    category_id = None
                try:
                    threshold = float(str(row.get("best_threshold") or "").strip())
                except ValueError:
                    threshold = None
                category = {0: "general", 4: "character", 9: "rating"}.get(category_id, "unknown")
                out.append(TagMetadata(name, category, category_id, threshold))
        return out

    def preprocess(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = min(self.pad_size / max(width, 1), self.pad_size / max(height, 1), 1.0)
        if scale < 1.0:
            rgb = rgb.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
        pad_x = self.pad_size - rgb.width
        pad_y = self.pad_size - rgb.height
        rgb = ImageOps.expand(rgb, (pad_x // 2, pad_y // 2, pad_x - pad_x // 2, pad_y - pad_y // 2), fill="white")
        rgb = rgb.resize((self.input_size, self.input_size), Image.Resampling.BICUBIC)
        left = max(0, (rgb.width - self.input_size) // 2)
        top = max(0, (rgb.height - self.input_size) // 2)
        rgb = rgb.crop((left, top, left + self.input_size, top + self.input_size))
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        array = (array - self.mean) / self.std
        return np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False)

    def load(self, model_dir: Path, *, device: str = "cpu") -> Dict[str, Any]:
        missing = self.validate_model_dir(model_dir)
        if missing:
            raise ModelLoadError(f"CAFormer model is incomplete; missing: {', '.join(missing)}")
        self._load_preprocess(model_dir)
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise ModelLoadError("ONNX Runtime is unavailable. Install onnxruntime to use CAFormer.") from exc
        try:
            session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise ModelLoadError(f"Could not load CAFormer ONNX model: {exc}") from exc
        inputs = session.get_inputs()
        if not inputs:
            raise ModelLoadError("CAFormer ONNX model has no input tensor.")
        outputs = {output.name.lower(): output.name for output in session.get_outputs()}
        output_name = outputs.get("prediction") or outputs.get("logits")
        if not output_name:
            raise ModelLoadError("CAFormer ONNX model is missing a prediction/logits output tensor.")
        return {
            "session": session,
            "input_name": inputs[0].name,
            "output_name": output_name,
            "output_is_probability": output_name == outputs.get("prediction"),
            "metadata": self.tag_metadata(model_dir),
            "device": "cpu",
        }

    def predict(self, loaded: Dict[str, Any], images: Sequence[Image.Image]) -> List[List[TagPrediction]]:
        tensor = np.stack([self.preprocess(image) for image in images], axis=0)
        values = loaded["session"].run([loaded["output_name"]], {loaded["input_name"]: tensor})[0]
        values = np.asarray(values, dtype=np.float32)
        scores = values if loaded.get("output_is_probability") else 1.0 / (1.0 + np.exp(-values))
        metadata = loaded["metadata"]
        return [[TagPrediction(meta.tag_key, float(score), meta.model_category, meta.recommended_threshold) for meta, score in zip(metadata, row)] for row in scores]
