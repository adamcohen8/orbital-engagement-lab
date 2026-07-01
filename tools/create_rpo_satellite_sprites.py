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
    iss_sprite = _iss_sprite()
    iss_sprite.save(ASSET_DIR / "rpo_iss_target_sprite.png")
    print(ASSET_DIR / "rpo_iss_target_sprite.png")
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


def _iss_sprite() -> Image.Image:
    size = SPRITE_SIZE_PX * RENDER_SCALE
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    glow_draw = ImageDraw.Draw(glow)

    truss = (205, 222, 232, 235)
    truss_shadow = (42, 54, 64, 230)
    module = (222, 236, 242, 246)
    module_shadow = (88, 105, 118, 230)
    panel = (26, 104, 155, 230)
    panel_edge = (96, 205, 242, 220)
    panel_grid = (10, 43, 70, 190)
    radiator = (196, 210, 214, 225)
    accent = TARGET_COLOR

    def p(x: float, y: float) -> tuple[int, int]:
        return int(round(x * RENDER_SCALE)), int(round(y * RENDER_SCALE))

    def line(points: tuple[tuple[float, float], ...], color: tuple[int, int, int, int], width: int = 2) -> None:
        draw.line([p(x, y) for x, y in points], fill=color, width=width * RENDER_SCALE, joint="curve")

    def poly(
        points: tuple[tuple[float, float], ...],
        outline: tuple[int, int, int, int],
        fill: tuple[int, int, int, int],
        width: int = 2,
    ) -> None:
        coords = [p(x, y) for x, y in points]
        draw.polygon(coords, fill=fill)
        draw.line(coords + [coords[0]], fill=outline, width=width * RENDER_SCALE, joint="curve")

    def panel_quad(points: tuple[tuple[float, float], ...], grid_axis: str) -> None:
        poly(points, panel_edge, panel, 2)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if grid_axis == "left":
            for x in (min_x + 8, min_x + 18, min_x + 28):
                line(((x, min_y + 2), (x - 4, max_y - 2)), panel_grid, 1)
        else:
            for x in (min_x + 8, min_x + 18, min_x + 28):
                line(((x, min_y + 2), (x + 4, max_y - 2)), panel_grid, 1)
        for y in (min_y + 7, min_y + 14):
            line(((min_x + 2, y), (max_x - 2, y + 3)), (90, 188, 226, 160), 1)

    glow_draw.ellipse((*p(42, 42), *p(86, 86)), outline=accent, width=7 * RENDER_SCALE)
    glow_draw.line((p(8, 64), p(120, 64)), fill=(80, 190, 235, 105), width=5 * RENDER_SCALE)
    image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(3 * RENDER_SCALE)))

    # ISS-like silhouette: a long truss with four solar array wings.
    line(((8, 66), (120, 62)), truss_shadow, 5)
    line(((9, 64), (119, 60)), truss, 3)
    for x in (21, 36, 51, 77, 92, 107):
        line(((x, 56), (x + 6, 69)), (118, 139, 150, 225), 1)
        line(((x, 69), (x + 6, 56)), (118, 139, 150, 190), 1)

    panel_quad(((8, 29), (42, 35), (41, 55), (7, 50)), "left")
    panel_quad(((7, 78), (41, 72), (42, 93), (8, 100)), "left")
    panel_quad(((86, 35), (121, 28), (121, 50), (87, 55)), "right")
    panel_quad(((87, 72), (121, 78), (120, 99), (86, 93)), "right")

    # Radiators and modules near the docking port.
    poly(((47, 42), (59, 44), (57, 58), (45, 56)), (224, 238, 244, 210), radiator, 1)
    poly(((70, 42), (83, 39), (82, 54), (70, 57)), (224, 238, 244, 210), radiator, 1)
    draw.rounded_rectangle((*p(47, 56), *p(81, 75)), radius=6 * RENDER_SCALE, fill=(24, 34, 42, 235), outline=module, width=2 * RENDER_SCALE)
    for x0, x1 in ((49, 59), (57, 68), (66, 78)):
        draw.rounded_rectangle((*p(x0, 58), *p(x1, 73)), radius=5 * RENDER_SCALE, fill=module, outline=module_shadow, width=1 * RENDER_SCALE)
    draw.ellipse((*p(39, 57), *p(53, 72)), fill=module, outline=module_shadow, width=1 * RENDER_SCALE)
    draw.ellipse((*p(75, 56), *p(89, 71)), fill=module, outline=module_shadow, width=1 * RENDER_SCALE)

    # Docking port and target reticle.
    draw.ellipse((*p(57, 55), *p(73, 71)), outline=accent, width=2 * RENDER_SCALE)
    draw.ellipse((*p(61, 59), *p(69, 67)), fill=(238, 249, 252, 245), outline=accent, width=1 * RENDER_SCALE)
    draw.line((p(65, 51), p(65, 75)), fill=accent, width=1 * RENDER_SCALE)
    draw.line((p(53, 63), p(77, 63)), fill=accent, width=1 * RENDER_SCALE)

    # Small antennas make the tiny icon feel less like a generic panel bus.
    line(((48, 56), (34, 45)), (232, 245, 248, 215), 1)
    line(((80, 57), (96, 44)), (232, 245, 248, 215), 1)
    draw.ellipse((*p(32, 43), *p(36, 47)), fill=(232, 245, 248, 220))
    draw.ellipse((*p(94, 42), *p(98, 46)), fill=(232, 245, 248, 220))

    return image.resize((SPRITE_SIZE_PX, SPRITE_SIZE_PX), Image.Resampling.LANCZOS)


def _preview_sheet(sprites: dict[str, tuple[int, int, int, int]]) -> Image.Image:
    sheet = Image.new("RGBA", (528, 190), (12, 16, 22, 255))
    draw = ImageDraw.Draw(sheet)
    preview = tuple(sprites.items()) + (("rpo_iss_target_sprite.png", TARGET_COLOR),)
    for idx, (filename, accent) in enumerate(preview):
        x = 32 + idx * 168
        draw.rounded_rectangle((x - 12, 18, x + 140, 170), radius=8, fill=(20, 27, 36, 255), outline=(80, 92, 110, 255), width=1)
        if (ASSET_DIR / filename).exists():
            sprite = Image.open(ASSET_DIR / filename).convert("RGBA")
        elif filename == "rpo_iss_target_sprite.png":
            sprite = _iss_sprite()
        else:
            sprite = _sprite(accent)
        sheet.alpha_composite(sprite, (x, 28))
        label = filename.replace("rpo_", "").replace("_sprite.png", "")
        draw.text((x + 2, 156), label, fill=(220, 234, 246, 255))
    return sheet


if __name__ == "__main__":
    main()
