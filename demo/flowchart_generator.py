"""
Synthesizes a simple architecture flowchart with Pillow at runtime, so the
demo needs no pre-encoded static asset file.

The diagram intentionally includes a "Redis Cache" block wired between the
Edge Gateway and the Cloud Dashboard that the default report text never
mentions — this is the planted discrepancy for Agent_VisionAlign to catch.
"""

from __future__ import annotations
from PIL import Image, ImageDraw, ImageFont


BOX_FILL = (235, 242, 255)
BOX_BORDER = (30, 62, 98)
DISCREPANCY_FILL = (255, 235, 235)
DISCREPANCY_BORDER = (178, 34, 34)
ARROW_COLOR = (60, 60, 60)
TEXT_COLOR = (20, 20, 20)
BG_COLOR = (255, 255, 255)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_box(draw: ImageDraw.ImageDraw, xy, label: str, font, fill=BOX_FILL, border=BOX_BORDER):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=10, fill=fill, outline=border, width=2)
    bbox = draw.textbbox((0, 0), label, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    draw.multiline_text((cx - w / 2, cy - h / 2), label, fill=TEXT_COLOR, font=font, align="center")


def _arrow(draw: ImageDraw.ImageDraw, p0, p1, color=ARROW_COLOR):
    draw.line([p0, p1], fill=color, width=3)
    import math

    angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    size = 8
    for da in (0.5, -0.5):
        a = angle + math.pi - da
        x = p1[0] + size * math.cos(a)
        y = p1[1] + size * math.sin(a)
        draw.line([p1, (x, y)], fill=color, width=3)


def generate_demo_flowchart() -> Image.Image:
    W, H = 1100, 460
    img = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font = _load_font(16)
    title_font = _load_font(20)

    draw.text((20, 15), "Edge-AI IoT Fall Detection — System Architecture", fill=TEXT_COLOR, font=title_font)

    boxes = {
        "sensor": (40, 180, 210, 250, "Sensor Node\n(Accel + Gyro)"),
        "gateway": (280, 180, 450, 250, "Edge Gateway\n(Raspberry Pi 4)"),
        "cnn": (520, 90, 690, 160, "Lightweight CNN\nInference (TFLite)"),
        "alert": (520, 270, 690, 340, "Alert Module\n(Local + Push)"),
        "redis": (760, 180, 930, 250, "Redis Cache\n(Event Buffer)"),
        "cloud": (960, 180, 1080, 250, "Cloud\nDashboard"),
    }

    for key, (x0, y0, x1, y1, label) in boxes.items():
        fill = DISCREPANCY_FILL if key == "redis" else BOX_FILL
        border = DISCREPANCY_BORDER if key == "redis" else BOX_BORDER
        _draw_box(draw, (x0, y0, x1, y1), label, font, fill=fill, border=border)

    def center_right(b):
        x0, y0, x1, y1, _ = boxes[b]
        return (x1, (y0 + y1) / 2)

    def center_left(b):
        x0, y0, x1, y1, _ = boxes[b]
        return (x0, (y0 + y1) / 2)

    def center_top(b):
        x0, y0, x1, y1, _ = boxes[b]
        return ((x0 + x1) / 2, y0)

    def center_bottom(b):
        x0, y0, x1, y1, _ = boxes[b]
        return ((x0 + x1) / 2, y1)

    _arrow(draw, center_right("sensor"), center_left("gateway"))
    _arrow(draw, center_top("gateway"), center_left("cnn"))
    _arrow(draw, center_bottom("gateway"), center_left("alert"))
    _arrow(draw, center_right("cnn"), center_left("redis"))
    _arrow(draw, center_right("alert"), center_left("redis"))
    _arrow(draw, center_right("redis"), center_left("cloud"))

    draw.text(
        (40, 400),
        "Note: 'Redis Cache' block is present in this diagram but is not mentioned anywhere\n"
        "in the report's written Proposed Architecture & Technical Route section (planted discrepancy).",
        fill=DISCREPANCY_BORDER,
        font=font,
    )

    return img


if __name__ == "__main__":
    img = generate_demo_flowchart()
    img.save("/tmp/demo_flowchart.png")
    print("saved /tmp/demo_flowchart.png")
