import math
from PIL import Image, ImageEnhance

def apply_preset(img: Image.Image, cfg: dict) -> Image.Image:
    ev = float(cfg.get("exposure_ev", 0))
    if ev != 0: img = ImageEnhance.Brightness(img).enhance(pow(2.0, ev))
    b = float(cfg.get("brightness", 0))
    if b != 0: img = ImageEnhance.Brightness(img).enhance(max(0.0, 1.0 + b))
    c = float(cfg.get("contrast", 0))
    if c != 0: img = ImageEnhance.Contrast(img).enhance(max(0.0, 1.0 + c))
    s = float(cfg.get("saturation", 0))
    if s != 0: img = ImageEnhance.Color(img).enhance(max(0.0, 1.0 + s))
    sh = float(cfg.get("sharpness", 0))
    if sh != 0: img = ImageEnhance.Sharpness(img).enhance(max(0.0, 1.0 + sh))

    # warmth [-1..1]
    w = float(cfg.get("warmth", 0))
    if w != 0:
        r,g,b = img.split()
        r = ImageEnhance.Brightness(r).enhance(1.0 + 0.5*max(0,w))
        bch = ImageEnhance.Brightness(b).enhance(1.0 - 0.5*max(0,w))
        if w < 0:
            r = ImageEnhance.Brightness(r).enhance(1.0 + 0.5*w)
            bch = ImageEnhance.Brightness(b).enhance(1.0 - 0.5*w)
        img = Image.merge("RGB", (r,g,bch))

    # tint [-1..1]
    t = float(cfg.get("tint", 0))
    if t != 0:
        r,g,b = img.split()
        g = ImageEnhance.Brightness(g).enhance(1.0 + 0.5*t)
        img = Image.merge("RGB", (r,g,b))

    # highlights/shadows
    hi = float(cfg.get("highlights", 0))
    shd = float(cfg.get("shadows", 0))
    if hi != 0 or shd != 0:
        def apply_curve(channel, curve):
            lut = [min(255, max(0, int(curve(i)))) for i in range(256)]
            return channel.point(lut)
        r,g,b = img.split()
        if shd != 0:
            def lift(x): xf = x/255.0; return 255.0 * (xf + shd * (1 - xf) * 0.5)
            r = apply_curve(r, lift); g = apply_curve(g, lift); b = apply_curve(b, lift)
        if hi != 0:
            def tame(x): xf = x/255.0; return 255.0 * (xf - hi * (xf**2) * 0.5)
            r = apply_curve(r, tame); g = apply_curve(g, tame); b = apply_curve(b, tame)
        img = Image.merge("RGB", (r,g,b))

    vig = float(cfg.get("vignette", 0))
    if vig > 0:
        w,h = img.size; cx,cy = w/2, h/2; maxd = math.hypot(cx, cy); px = img.load()
        for y in range(h):
            for x in range(w):
                d = math.hypot(x-cx, y-cy) / maxd
                factor = 1 - vig * (d**2)
                R,G,B = px[x,y]; px[x,y] = (int(R*factor), int(G*factor), int(B*factor))
    return img
