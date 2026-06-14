from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "sim" / "game" / "assets"
SPRITE_SIZE_PX = 128
RENDER_SCALE = 4

TARGET_COLOR = (245, 92, 92, 255)
CHASER_COLOR = (245, 205, 92, 255)
PANEL_COLOR = (70, 190, 245, 210)
CORE_DIM = (80, 110, 130, 180)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    sprites = {
        "rpo_target_sprite.png": TARGET_COLOR,
        "rpo_chaser_sprite.png": CHASER_COLOR,
    }
    for filename, accent in sprites.items():
        image = _sprite(accent)
        image.save(ASSET_DIR / filename)
        print(ASSET_DIR / filename)
    _preview_sheet(sprites).save(ASSET_DIR / "rpo_satellite_sprites_preview.png")
    print(ASSET_DIR / "rpo_satellite_sprites_preview.png")


def _sprite(accent: tuple[int, int, int, int]) -> Image.Image:
    size = SPRITE_SIZE_PX * RENDER_SCALE
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    glow_draw = ImageDraw.Draw(glow)

    def p(x: float, y: float) -> tuple[int, int]:
        return int(round(x * RENDER_SCALE)), int(round(y * RENDER_SCALE))

    def line(points: tuple[tuple[float, float], ...], color: tuple[int, int, int, int], width: int = 2) -> None:
        coords = [p(x, y) for x, y in points]
        draw.line(coords, fill=color, width=width * RENDER_SCALE, joint="curve")

    def poly(points: tuple[tuple[float, float], ...], outline: tuple[int, int, int, int], fill: tuple[int, int, int, int]) -> None:
        coords = [p(x, y) for x, y in points]
        draw.polygon(coords, fill=fill)
        draw.line(coords + [coords[0]], fill=outline, width=2 * RENDER_SCALE, joint="curve")

    glow_draw.ellipse((*p(37, 37), *p(91, 91)), outline=accent, width=6 * RENDER_SCALE)
    glow_draw.line((p(11, 83), p(51, 68), p(77, 59), p(119, 45)), fill=accent, width=4 * RENDER_SCALE)
    glow = glow.filter(ImageFilter.GaussianBlur(4 * RENDER_SCALE))
    image.alpha_composite(glow)

    # Bus and service module.
    draw.rounded_rectangle((*p(49, 48), *p(79, 80)), radius=8 * RENDER_SCALE, fill=(8, 18, 26, 235), outline=accent, width=2 * RENDER_SCALE)
    draw.ellipse((*p(54, 53), *p(74, 73)), outline=accent, width=2 * RENDER_SCALE)
    draw.arc((*p(57, 56), *p(71, 70)), start=35, end=325, fill=(220, 242, 250, 220), width=1 * RENDER_SCALE)
    line(((49, 58), (79, 58)), CORE_DIM, 1)
    line(((49, 69), (79, 69)), CORE_DIM, 1)

    # Solar arrays.
    poly(((18, 38), (48, 51), (44, 78), (14, 65)), PANEL_COLOR, (8, 30, 42, 190))
    poly(((80, 51), (112, 64), (106, 91), (76, 78)), PANEL_COLOR, (8, 30, 42, 190))
    for x0, y0, x1, y1 in ((23, 40, 19, 65), (32, 44, 29, 70), (41, 48, 38, 74), (85, 53, 81, 80), (94, 57, 90, 84), (103, 61, 99, 88)):
        line(((x0, y0), (x1, y1)), (46, 126, 174, 180), 1)
    line(((16, 51), (45, 64)), (46, 126, 174, 180), 1)
    line(((79, 64), (109, 77)), (46, 126, 174, 180), 1)

    # Boom/antenna and exact-position reticle.
    line(((7, 91), (50, 72)), accent, 2)
    line(((78, 56), (122, 41)), accent, 2)
    draw.ellipse((*p(61, 61), *p(67, 67)), fill=(235, 248, 255, 240), outline=accent, width=1 * RENDER_SCALE)
    draw.line((p(64, 56), p(64, 72)), fill=accent, width=1 * RENDER_SCALE)
    draw.line((p(56, 64), p(72, 64)), fill=accent, width=1 * RENDER_SCALE)

    return image.resize((SPRITE_SIZE_PX, SPRITE_SIZE_PX), Image.Resampling.LANCZOS)


def _preview_sheet(sprites: dict[str, tuple[int, int, int, int]]) -> Image.Image:
    sheet = Image.new("RGBA", (360, 190), (12, 16, 22, 255))
    draw = ImageDraw.Draw(sheet)
    for idx, (filename, _) in enumerate(sprites.items()):
        x = 32 + idx * 168
        draw.rounded_rectangle((x - 12, 18, x + 140, 170), radius=8, fill=(20, 27, 36, 255), outline=(80, 92, 110, 255), width=1)
        sprite = Image.open(ASSET_DIR / filename).convert("RGBA") if (ASSET_DIR / filename).exists() else _sprite(sprites[filename])
        sheet.alpha_composite(sprite, (x, 28))
        draw.text((x + 2, 156), filename.replace("rpo_", "").replace("_sprite.png", ""), fill=(220, 234, 246, 255))
    return sheet


if __name__ == "__main__":
    main()
