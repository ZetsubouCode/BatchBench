from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "static" / "icons" / "icon.ico"
OUTPUT = ROOT / "static" / "icons" / "discord_presence.png"


def _largest_ico_frame(path: Path) -> Image.Image:
    with Image.open(path) as image:
        if hasattr(image, "ico") and hasattr(image.ico, "sizes"):
            sizes = sorted(image.ico.sizes(), key=lambda size: size[0] * size[1])
            if sizes:
                return image.ico.getimage(sizes[-1]).convert("RGBA")

        frames = []
        frame_count = getattr(image, "n_frames", 1)
        for index in range(frame_count):
            try:
                image.seek(index)
                frames.append(image.convert("RGBA").copy())
            except EOFError:
                break
        if not frames:
            return image.convert("RGBA")
        return max(frames, key=lambda frame: frame.width * frame.height)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Missing icon: {SOURCE.relative_to(ROOT)}")

    icon = _largest_ico_frame(SOURCE)
    icon.thumbnail((512, 512), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    x = (512 - icon.width) // 2
    y = (512 - icon.height) // 2
    canvas.alpha_composite(icon, (x, y))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT)

    print("Generated Discord Presence asset:")
    print(OUTPUT.relative_to(ROOT).as_posix())
    print("Size: 512x512")


if __name__ == "__main__":
    main()
