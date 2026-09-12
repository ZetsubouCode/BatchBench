from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, List, Sequence

from PIL import Image

from .base import ModelLoadError, TagMetadata, TagPrediction, TaggerAdapter


class SmilingWolfWdAdapter(TaggerAdapter):
    family = "smilingwolf_wd"
    preprocessing_label = "smilingwolf_wd (BGR)"
    channel_order = "BGR"

    def required_files(self) -> Sequence[str]:
        return ("model.safetensors", "config.json", "selected_tags.csv")

    def tag_metadata(self, model_dir: Path) -> List[TagMetadata]:
        path = model_dir / "selected_tags.csv"
        if not path.is_file():
            return []
        out: List[TagMetadata] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                name = str(row.get("name") or row.get("tag") or "").strip()
                if not name:
                    continue
                try:
                    category_id = int(str(row.get("category") or "").strip())
                except ValueError:
                    category_id = None
                category = {0: "general", 1: "artist", 2: "copyright", 3: "character", 4: "meta", 9: "rating"}.get(category_id, "unknown")
                out.append(TagMetadata(name, category, category_id))
        return out

    def preprocess(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        red, green, blue = rgb.split()
        return Image.merge("RGB", (blue, green, red))

    def load(self, model_dir: Path, *, device: str = "cpu") -> Any:
        missing = self.validate_model_dir(model_dir)
        if missing:
            raise ModelLoadError(f"WD model is incomplete; missing: {', '.join(missing)}")
        try:
            import numpy as np
            import torch
            import timm
            from safetensors.torch import load_file
        except Exception as exc:
            raise ModelLoadError(f"SmilingWolf runtime unavailable: {exc}") from exc
        try:
            config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
            architecture = str(config["architecture"])
            num_classes = int(config["num_classes"])
            model_args = dict(config.get("model_args") or {})
            model_args["num_classes"] = num_classes
            model = timm.create_model(architecture, pretrained=False, **model_args)
            model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
        except Exception as exc:
            raise ModelLoadError(f"Could not load WD Timm model: {exc}") from exc
        resolved = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        model.eval().to(resolved)
        preprocessing = dict(config.get("pretrained_cfg") or {})
        input_size = preprocessing.get("input_size") or [3, 448, 448]
        return {
            "model": model,
            "device": resolved,
            "torch": torch,
            "numpy": np,
            "input_size": int(input_size[-1]),
            "mean": tuple(float(value) for value in (preprocessing.get("mean") or [0.5, 0.5, 0.5])),
            "std": tuple(float(value) for value in (preprocessing.get("std") or [0.5, 0.5, 0.5])),
            "interpolation": str(preprocessing.get("interpolation") or "bicubic"),
            "metadata": self.tag_metadata(model_dir),
        }

    def predict(self, loaded: Any, images: Sequence[Image.Image]) -> List[List[TagPrediction]]:
        np = loaded["numpy"]
        size = loaded["input_size"]
        mean = np.asarray(loaded["mean"], dtype=np.float32)
        std = np.asarray(loaded["std"], dtype=np.float32)
        interpolation = Image.Resampling.BICUBIC if loaded["interpolation"] == "bicubic" else Image.Resampling.BILINEAR
        prepared = []
        for image in images:
            converted = self.preprocess(image)
            side = max(converted.size)
            square = Image.new("RGB", (side, side), (255, 255, 255))
            square.paste(converted, ((side - converted.width) // 2, (side - converted.height) // 2))
            array = np.asarray(square.resize((size, size), interpolation), dtype=np.float32) / 255.0
            prepared.append(np.transpose((array - mean) / std, (2, 0, 1)))
        inputs = loaded["torch"].from_numpy(np.stack(prepared)).to(loaded["device"])
        with loaded["torch"].no_grad():
            values = loaded["torch"].sigmoid(loaded["model"](inputs)).cpu().numpy()
        metadata = loaded["metadata"]
        return [[TagPrediction(meta.tag_key, float(score), meta.model_category) for meta, score in zip(metadata, row)] for row in values]
