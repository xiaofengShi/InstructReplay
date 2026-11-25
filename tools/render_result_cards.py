"""Render the README charts and social preview with Matplotlib."""

from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"
INK = "#17243d"
MUTED = "#4b5c77"
PAPER = "#f4f7fd"
BORDER = "#d9e3f2"
BASELINE = "#7e91ad"
REPLAY = "#2455d8"

# Published means retain the paper's precision. Per-setting CSV rows are rounded.
METRICS = (
    ("General forgetting", "Lower is better", 28.07, 2.44),
    ("Domain acquisition", "Higher is better", 15.23, 15.49),
)
SCALE = 30
DPI = 100


def draw_card(filename, width, height, *, stacked=False, social=False):
    figure = Figure(figsize=(width / DPI, height / DPI), dpi=DPI, facecolor="none")
    canvas = figure.add_axes((0, 0, 1, 1), xlim=(0, width), ylim=(height, 0), frame_on=False)
    canvas.set_axis_off()

    def box(x, y, w, h, *, fill, radius, stroke="none"):
        canvas.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=0.85, edgecolor=stroke, facecolor=fill,
        ))

    def text(x, y, value, pixels, *, color=INK, bold=False, align="left"):
        canvas.text(
            x, y, value, color=color, fontfamily="DejaVu Sans",
            fontsize=pixels * 72 / DPI, fontweight="bold" if bold else "normal",
            ha=align, va="center",
        )

    box(1, 1, width - 2, height - 2, fill=PAPER, radius=24, stroke=BORDER)
    if social:
        text(40, 70, "InstructReplay", 62, bold=True)
        text(42, 138, "Factual fine-tuning with less forgetting", 36)
        text(42, 195, "arXiv:2611.32918", 26, color=REPLAY, bold=True)
        text(42, 567, "Mean of six settings · 3 model families · 2 domains", 25, color=MUTED)
    else:
        text(32, 42, "Main results", 34, bold=True)
        x = 32 if stacked else width - 32
        y = 78 if stacked else 43
        text(x, y, "Mean across six settings · percentage points", 23, color=MUTED,
             align="left" if stacked else "right")

    margin, gap, panel_height = 32, 20, 230
    panel_width = width - 2 * margin if stacked else (width - 2 * margin - gap) / 2
    top = 260 if social else 109 if stacked else 86
    for index, (label, direction, baseline, replay) in enumerate(METRICS):
        x = margin if stacked else margin + index * (panel_width + gap)
        y = top + index * (panel_height + gap) if stacked else top
        box(x, y, panel_width, panel_height, fill="#ffffff", radius=19, stroke=BORDER)
        text(x + 22, y + 35, label, 27, bold=True)
        text(x + 22, y + 68, direction, 20, color=MUTED)

        start = x + 196
        end = x + panel_width - 110
        span = end - start
        for row, (method, value, color) in enumerate((
            ("Naive SFT", baseline, BASELINE),
            ("InstructReplay", replay, REPLAY),
        )):
            line = y + 116 + row * 45
            active = row == 1
            text(x + 22, line, method, 19, bold=active)
            box(start, line - 6, span, 12, fill=PAPER, radius=6)
            box(start, line - 6, span * value / SCALE, 12, fill=color, radius=6)
            text(x + panel_width - 22, line, f"{value:.2f}", 24,
                 color=REPLAY if active else INK, bold=active, align="right")

        text(start, y + 199, "0", 17, color=MUTED)
        text(end, y + 199, f"{SCALE} pp", 17, color=MUTED, align="right")

    figure.savefig(ASSETS / filename, dpi=DPI, transparent=True)


if __name__ == "__main__":
    draw_card("readme-results.png", 1200, 349)
    draw_card("readme-results-mobile.png", 620, 621, stacked=True)
    draw_card("result-card.png", 1200, 630, social=True)
