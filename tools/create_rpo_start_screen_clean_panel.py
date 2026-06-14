from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sim" / "game" / "assets" / "OEL_RPO_Trainer.png"
OUTPUT = SOURCE


CYAN = (0, 190, 255)
CYAN_DIM = (0, 82, 128)
CYAN_FAINT = (0, 50, 84)
BORDER_CYAN = (10, 110, 153)
BORDER_SIDE = (4, 61, 92)
WHITE_BLUE = (200, 244, 255)
GOLD = (255, 222, 44)
ORANGE = (255, 104, 40)
BLUE = (50, 210, 255)
PANEL_FILL = (2, 8, 14)


def main() -> None:
    image = Image.open(SOURCE).convert("RGB")
    draw = ImageDraw.Draw(image)
    _replace_bottom_left_panel(draw)
    image.save(OUTPUT)
    print(OUTPUT)


def _replace_bottom_left_panel(draw: ImageDraw.ImageDraw) -> None:
    panel = [(51, 759), (386, 759), (410, 783), (410, 880), (387, 903), (51, 903), (29, 881), (29, 781)]
    draw.rectangle((24, 752, 414, 907), fill=(0, 0, 0))
    shadow = [(x + 2, y + 2) for x, y in panel]
    draw.polygon(shadow, fill=(0, 4, 8))
    draw.polygon(panel, fill=PANEL_FILL)
    draw.line(panel + [panel[0]], fill=BORDER_SIDE, width=3)
    draw.line(panel + [panel[0]], fill=BORDER_CYAN, width=1)

    # Preserve the long decorative lower HUD rail, but keep the new module edge clean.
    draw.line((48, 916, 242, 916), fill=BORDER_CYAN, width=3)
    draw.line((255, 916, 345, 916), fill=CYAN_DIM, width=2)

    _draw_scope(draw)


def _draw_scope(draw: ImageDraw.ImageDraw) -> None:
    _draw_spacecraft_glyph(draw, center=(168, 830), scale=1.38)
    _draw_meter_blocks(draw, origin=(295, 790))


def _draw_spacecraft_glyph(draw: ImageDraw.ImageDraw, *, center: tuple[int, int], scale: float) -> None:
    cx, cy = center

    def point(dx: float, dy: float) -> tuple[int, int]:
        return (int(round(cx + dx * scale)), int(round(cy + dy * scale)))

    draw.line((point(-82, 28), point(-22, 8)), fill=CYAN, width=2)
    draw.line((point(22, -8), point(86, -28)), fill=CYAN, width=2)
    body = (*point(-22, -12), *point(22, 12))
    draw.rounded_rectangle(body, radius=8, outline=CYAN, width=2)
    draw.ellipse((*point(-15, -9), *point(15, 9)), outline=CYAN_DIM, width=1)
    draw.arc((*point(-14, -8), *point(14, 8)), start=25, end=335, fill=CYAN, width=1)
    draw.line((point(-22, -3), point(22, -3)), fill=CYAN_DIM, width=1)
    draw.line((point(-22, 4), point(22, 4)), fill=CYAN_DIM, width=1)
    draw.line((point(-43, -4), point(-22, -4)), fill=CYAN, width=2)
    draw.line((point(22, 4), point(44, 4)), fill=CYAN, width=2)

    left_panel = [point(-68, -29), point(-40, -17), point(-34, 6), point(-62, -6)]
    right_panel = [point(42, 0), point(72, 12), point(65, 34), point(36, 20)]
    draw.line(left_panel + [left_panel[0]], fill=CYAN, width=2)
    draw.line(right_panel + [right_panel[0]], fill=CYAN, width=2)
    for offset in (-59, -49):
        draw.line((point(offset, -25), point(offset + 8, -2)), fill=CYAN_DIM, width=1)
    for offset in (48, 59):
        draw.line((point(offset, 4), point(offset + 8, 27)), fill=CYAN_DIM, width=1)
    draw.ellipse((*point(-5, -5), *point(5, 5)), outline=CYAN, width=1)


def _draw_meter_blocks(draw: ImageDraw.ImageDraw, *, origin: tuple[int, int]) -> None:
    x, y = origin
    columns = (
        (0, 0, 0, 1, 1, 1),
        (0, 1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 1),
        (0, 0, 1, 1, 2, 2),
    )
    colors = {
        0: (7, 26, 38),
        1: (0, 150, 218),
        2: ORANGE,
    }
    for col, rows in enumerate(columns):
        for row, state in enumerate(rows):
            x0 = x + col * 13
            y0 = y + row * 13
            draw.rectangle((x0, y0, x0 + 8, y0 + 7), fill=colors[state])


if __name__ == "__main__":
    main()
