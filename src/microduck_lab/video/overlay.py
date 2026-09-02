"""Text/UI overlays for the hero video (PIL compositing on rendered frames)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_CJK_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def _font(size: int, bold=True, cjk=False):
    paths = (_CJK_PATHS + _FONT_PATHS) if cjk else _FONT_PATHS
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_text(draw, xy, text, size=28, fill=(255, 255, 255, 255), anchor="la",
               shadow=True, cjk=False):
    f = _font(size, cjk=cjk)
    if shadow:
        draw.text((xy[0] + 2, xy[1] + 2), text, font=f, fill=(0, 0, 0, 200), anchor=anchor)
    draw.text(xy, text, font=f, fill=fill, anchor=anchor)


def overlay(img: np.ndarray, *, status: str | None = None, chat: str | None = None,
            chat_answer: str | None = None, title: str | None = None,
            metrics: list[str] | None = None, subtitle: str | None = None) -> np.ndarray:
    """Compose UI text on a rendered frame. Returns new RGB array."""
    im = Image.fromarray(img).convert("RGBA")
    W, H = im.size
    draw = ImageDraw.Draw(im)

    if title:
        # left-aligned, clear of the top-right metrics panel
        _draw_text(draw, (170, 24), title, size=30, anchor="la")
    if subtitle:
        _draw_text(draw, (170, 62), subtitle, size=20, anchor="la",
                   fill=(255, 235, 150, 255), cjk=True)

    if status:
        colors = {"THINKING": (150, 180, 255), "ACTING": (120, 220, 130),
                  "PRACTICING": (255, 200, 90), "EVOLVING": (255, 140, 220),
                  "CHAMPION": (255, 220, 80)}
        col = colors.get(status, (255, 255, 255))
        # chip
        f = _font(24)
        tw = draw.textlength(status, font=f)
        x0, y0 = 18, 18
        draw.rounded_rectangle([x0 - 10, y0 - 6, x0 + tw + 22, y0 + 34], 12,
                               fill=(20, 24, 32, 210), outline=col + (255,), width=2)
        draw.ellipse([x0 + 2, y0 + 10, x0 + 14, y0 + 22], fill=col + (255,))
        draw.text((x0 + 20, y0 + 3), status, font=f, fill=(255, 255, 255, 255))

    if chat:
        # user bubble bottom-left
        f = _font(26, cjk=True)
        tw = draw.textlength(chat, font=f)
        bx0, by0 = 18, H - 96
        draw.rounded_rectangle([bx0, by0, bx0 + tw + 36, by0 + 44], 14,
                               fill=(60, 110, 220, 235))
        draw.text((bx0 + 18, by0 + 8), chat, font=f, fill=(255, 255, 255, 255))
    if chat_answer:
        f = _font(24, cjk=True)
        bx0, by0 = 18, H - 46
        draw.rounded_rectangle([bx0, by0, bx0 + draw.textlength(chat_answer, font=f) + 36, by0 + 40],
                               14, fill=(40, 44, 54, 235))
        draw.text((bx0 + 18, by0 + 7), chat_answer, font=f, fill=(180, 255, 190, 255))

    if metrics:
        # metrics panel top-right
        f = _font(20)
        lines = metrics
        wmax = max(draw.textlength(l, font=f) for l in lines)
        px1, py1 = W - 18, 18
        px0 = px1 - wmax - 36
        py0 = py1
        draw.rounded_rectangle([px0, py0, px1, py1 + 26 * len(lines) + 18], 10,
                               fill=(15, 18, 26, 215))
        for i, l in enumerate(lines):
            draw.text((px0 + 14, py0 + 10 + 26 * i), l, font=f, fill=(220, 228, 255, 255))

    return np.asarray(im.convert("RGB"))


def title_card(text: str, sub: str = "", size=(960, 540), color=(16, 20, 30)) -> np.ndarray:
    im = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(im)
    _draw_text(draw, (size[0] // 2, size[1] // 2 - 40), text, size=52, anchor="mm", shadow=False)
    if sub:
        _draw_text(draw, (size[0] // 2, size[1] // 2 + 24), sub, size=26, anchor="mm",
                   fill=(255, 220, 120), shadow=False, cjk=True)
    return np.asarray(im)


def grid_montage(frames: list[np.ndarray], cols: int = 4, pad: int = 6,
                 bg=(15, 18, 25)) -> np.ndarray:
    """Tile frames into a grid (practice montage)."""
    if not frames:
        raise ValueError("empty")
    h, w = frames[0].shape[:2]
    rows = (len(frames) + cols - 1) // cols
    W = cols * w + (cols + 1) * pad
    H = rows * h + (rows + 1) * pad
    im = Image.new("RGB", (W, H), bg)
    for i, f in enumerate(frames):
        tile = Image.fromarray(f)
        x = pad + (i % cols) * (w + pad)
        y = pad + (i // cols) * (h + pad)
        im.paste(tile, (x, y))
    return np.asarray(im)
