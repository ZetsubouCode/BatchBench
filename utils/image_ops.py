import numpy as np
from PIL import Image, ImageEnhance

# Slider inputs follow Windows Photos semantics: [-1, 1] ~= [-100, 100].


def _clamp(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(val)))
    except Exception:
        return 0.0


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def _apply_exposure(arr: np.ndarray, ev: float) -> np.ndarray:
    if ev == 0:
        return arr
    return arr * pow(2.0, ev)


def _apply_contrast(arr: np.ndarray, contrast: float) -> np.ndarray:
    if contrast == 0:
        return arr
    pivot = 0.5
    factor = max(0.4, 1.0 + 0.6 * contrast)  # gentle range, avoid inversion
    return (arr - pivot) * factor + pivot


def _apply_brightness(arr: np.ndarray, brightness: float) -> np.ndarray:
    if brightness == 0:
        return arr
    # Midtone-weighted lift so highlights retain detail. Range capped to avoid washout.
    mid_weight = 1.0 - np.abs(arr - 0.5) * 2.0
    delta = brightness * 0.22 * (0.5 + 0.5 * mid_weight)
    return arr + delta


def _apply_highlights_shadows(arr: np.ndarray, highlights: float, shadows: float) -> np.ndarray:
    if highlights == 0 and shadows == 0:
        return arr
    lum = arr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    out = arr
    if highlights != 0:
        mask = np.clip((lum - 0.5) / 0.5, 0.0, 1.0)
        adj = highlights * 0.35
        if highlights > 0:
            out = out + adj * mask[..., None] * (1.0 - out)
        else:
            out = out - (-adj) * mask[..., None] * out
    if shadows != 0:
        mask = np.clip((0.5 - lum) / 0.5, 0.0, 1.0)
        adj = shadows * 0.35
        if shadows > 0:
            out = out + adj * mask[..., None] * (1.0 - out)
        else:
            out = out - (-adj) * mask[..., None] * out
    return out


def _apply_white_balance(arr: np.ndarray, warmth: float, tint: float) -> np.ndarray:
    if warmth == 0 and tint == 0:
        return arr
    r_gain = 1.0 + 0.2 * warmth - 0.08 * tint
    g_gain = 1.0 + 0.04 * warmth + 0.2 * tint
    b_gain = 1.0 - 0.2 * warmth - 0.08 * tint
    gains = np.array([r_gain, g_gain, b_gain], dtype=np.float32)
    gains /= np.mean(gains)
    return arr * gains


def _apply_saturation(arr: np.ndarray, saturation: float) -> np.ndarray:
    if saturation == 0:
        return arr
    lum = arr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    if saturation >= 0:
        factor = 1.0 + saturation
    else:
        factor = 1.0 + saturation * 0.85
    return lum[..., None] + (arr - lum[..., None]) * factor


def _apply_vignette(arr: np.ndarray, vig: float) -> np.ndarray:
    if vig <= 0:
        return arr
    h, w, _ = arr.shape
    y, x = np.ogrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    norm = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / np.sqrt(cx**2 + cy**2)
    mask = 1.0 - vig * (norm**2)
    mask = np.clip(mask, 0.0, 1.0)[..., None]
    return arr * mask


def apply_preset(img: Image.Image, cfg: dict) -> Image.Image:
    ev = _clamp(cfg.get("exposure_ev", 0), -4.0, 4.0)
    brightness = _clamp(cfg.get("brightness", 0))
    contrast = _clamp(cfg.get("contrast", 0))
    highlights = _clamp(cfg.get("highlights", 0))
    shadows = _clamp(cfg.get("shadows", 0))
    saturation = _clamp(cfg.get("saturation", 0))
    warmth = _clamp(cfg.get("warmth", 0))
    tint = _clamp(cfg.get("tint", 0))
    sharpness = _clamp(cfg.get("sharpness", 0))
    vignette = max(0.0, float(cfg.get("vignette", 0)))

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

    arr = _clip01(_apply_exposure(arr, ev))
    arr = _clip01(_apply_brightness(arr, brightness))
    arr = _clip01(_apply_contrast(arr, contrast))
    arr = _clip01(_apply_highlights_shadows(arr, highlights, shadows))
    arr = _clip01(_apply_saturation(arr, saturation))
    arr = _clip01(_apply_white_balance(arr, warmth, tint))
    arr = _clip01(_apply_vignette(arr, vignette))

    out = Image.fromarray((arr * 255.0 + 0.5).astype("uint8"), mode="RGB")

    if sharpness != 0:
        out = ImageEnhance.Sharpness(out).enhance(max(0.0, 1.0 + sharpness * 1.5))
    return out
