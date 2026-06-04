#!/usr/bin/env python3
"""from fig2_task2_extend_metrics_merged.csv MAE fig ( celltype, stdlib → SVG + PNG)."""
import csv
import re
import struct
import zlib
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "fig2_task2_extend_metrics_merged.csv"
OUT_SVG = Path(__file__).resolve().parent / "mae_bar_temp.svg"
OUT_PNG = Path(__file__).resolve().parent / "mae_bar_temp.png"

# only typescell: "B" or "NK"
DATASET = "B"

# figdefault with Squidiff (MAE and amountlevel); need 6 canin order keepandinunder 
INCLUDE_SQUIDIFF = False

# , and (0.65–0.92)
BAR_WIDTH_FRAC = 0.88


def short_method(name: str) -> str:
    n = str(name).strip()
    if n.startswith("scrna_ddpm"):
        return "DDPM"
    if n.startswith("mlp_ddpm"):
        return "DDPM+MLP"
    if n.startswith("scDiffusion"):
        return "scDiffusion"
    if n == "Squidiff":
        return "Squidiff"
    if n.startswith("scDiff"):
        return "scDiff"
    if n == "scGen":
        return "scGen"
    return n


def parse_mae(s) -> tuple[float, float]:
    s = str(s).strip()
    m = re.match(r"^([\d.eE+-]+)\s*±\s*([\d.eE+-]+)$", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return float(s), 0.0


METHOD_ORDER_FULL = [
    "DDPM",
    "DDPM+MLP",
    "scGen",
    "scDiff",
    "scDiffusion",
    "Squidiff",
]


def load_rows():
    rows = []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for line in r:
            rows.append(
                {
                    "Dataset": line["Dataset"].strip(),
                    "method": short_method(line["Method"]),
                    "mae": parse_mae(line["MAE (mean±std)"])[0],
                }
            )
    return rows


def schematic_svg(
    values: list[tuple[str, float]],
    width: float = 220,
    height: float = 140,
    pad: float = 6,
) -> str:
    """ , ; only .values: (method_id, mae) → ."""
    maes = [v for _, v in values]
    n = len(values)
    if n == 0:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"/>'

    plot_w = width - 2 * pad
    plot_h = height - 2 * pad

    ymax = max(maes) * 1.05
    ymin = 0.0
    if ymax <= 0:
        ymax = 1.0

    slot_w = plot_w / n
    bar_w = slot_w * BAR_WIDTH_FRAC

    # distinguish methods ( fig , only )
    palette = [
        "#6B8EB7",
        "#8FAD8A",
        "#C89B7B",
        "#9B8AB8",
        "#B7A36B",
        "#7BA8A8",
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#fafafa"/>',
    ]

    for i, (_m, mu) in enumerate(values):
        cx_slot = pad + i * slot_w + slot_w / 2
        x0 = cx_slot - bar_w / 2
        t = (mu - ymin) / (ymax - ymin) if ymax > ymin else 0.0
        bar_h = t * plot_h
        y0 = pad + plot_h - bar_h
        col = palette[i % len(palette)]
        parts.append(
            f'<rect x="{x0:.3f}" y="{y0:.3f}" width="{bar_w:.3f}" height="{bar_h:.3f}" '
            f'fill="{col}" rx="1.5" ry="1.5"/>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def schematic_png_bytes(
    values: list[tuple[str, float]],
    width: int = 196,
    height: int = 120,
    pad: float = 4,
) -> bytes:
    """and schematic_svg , as RGB PNG ."""
    bg = _hex_rgb("#fafafa")
    maes = [v for _, v in values]
    n = len(values)
    if n == 0:
        buf = bytearray([bg[0], bg[1], bg[2]] * (width * height))
    else:
        plot_w = width - 2 * pad
        plot_h = height - 2 * pad
        ymax = max(maes) * 1.05
        ymin = 0.0
        if ymax <= 0:
            ymax = 1.0
        slot_w = plot_w / n
        bar_w = slot_w * BAR_WIDTH_FRAC
        palette = [
            "#6B8EB7",
            "#8FAD8A",
            "#C89B7B",
            "#9B8AB8",
            "#B7A36B",
            "#7BA8A8",
        ]
        buf = bytearray([bg[0], bg[1], bg[2]] * (width * height))

        def fill_rect(x0: int, y0: int, x1: int, y1: int, r: int, g: int, b: int) -> None:
            x0 = max(0, min(width, x0))
            x1 = max(0, min(width, x1))
            y0 = max(0, min(height, y0))
            y1 = max(0, min(height, y1))
            for yy in range(y0, y1):
                row = yy * width * 3
                for xx in range(x0, x1):
                    j = row + xx * 3
                    buf[j] = r
                    buf[j + 1] = g
                    buf[j + 2] = b

        for i, (_m, mu) in enumerate(values):
            cx_slot = pad + i * slot_w + slot_w / 2
            x0f = cx_slot - bar_w / 2
            t = (mu - ymin) / (ymax - ymin) if ymax > ymin else 0.0
            bar_h = t * plot_h
            y0f = pad + plot_h - bar_h
            r, g, b = _hex_rgb(palette[i % len(palette)])
            ix0 = int(round(x0f))
            ix1 = int(round(x0f + bar_w))
            iy0 = int(round(y0f))
            iy1 = int(round(y0f + bar_h))
            fill_rect(ix0, iy0, ix1, iy1, r, g, b)

    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        row = buf[y * stride : (y + 1) * stride]
        raw.extend(row)
    compressed = zlib.compress(bytes(raw), 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    out = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    return out


def main() -> None:
    rows = load_rows()
    by_m: dict[str, float] = {}
    for r in rows:
        if r["Dataset"] != DATASET:
            continue
        by_m[r["method"]] = r["mae"]

    if INCLUDE_SQUIDIFF:
        order = METHOD_ORDER_FULL
    else:
        order = [m for m in METHOD_ORDER_FULL if m != "Squidiff"]
    values = [(m, by_m[m]) for m in order if m in by_m]

    svg = schematic_svg(values, width=196, height=120, pad=4)
    OUT_SVG.write_text(svg, encoding="utf-8")
    OUT_PNG.write_bytes(schematic_png_bytes(values, width=196, height=120, pad=4))
    print(f"Wrote {OUT_SVG} and {OUT_PNG} ({DATASET} cells, {len(values)} bars)")


if __name__ == "__main__":
    main()
