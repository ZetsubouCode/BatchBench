from PIL import Image

from utils.image_ops import apply_preset


def test_warmth_slider_is_gradual():
    img = Image.new("RGB", (1, 1), (120, 130, 140))

    cool = apply_preset(img, {"warmth": -0.5}).getpixel((0, 0))
    neutral = apply_preset(img, {"warmth": 0.0}).getpixel((0, 0))
    mild = apply_preset(img, {"warmth": 0.25}).getpixel((0, 0))
    warm = apply_preset(img, {"warmth": 0.5}).getpixel((0, 0))

    assert cool != neutral
    assert neutral != mild
    assert mild != warm
    assert cool[0] < neutral[0] < mild[0] < warm[0]
    assert cool[2] > neutral[2] > mild[2] > warm[2]
