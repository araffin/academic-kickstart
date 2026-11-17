#!/usr/bin/env python3
"""
Convert ex-based width/height attributes in SVG files to fixed px values.

Excalidraw/MathJax SVG exports size their formulas in `ex` units, which resolve
against the font metrics of the rendering browser. On a browser with different
metrics, the formulas shift and can overflow the export frame, getting clipped
by the SVG viewport. This script rewrites each `width`/`height` attribute
expressed in `ex` to the fixed px value the exporter used (1ex = font-size / 2,
typically 10px for the 20px font of excalidraw exports), making the rendering
browser-independent.

Usage:
    python3 fix_ex_units.py [path ...]

Paths may be files or directories (scanned recursively). With no arguments,
all .svg files under the script's directory are processed.
"""

import re
import sys
from pathlib import Path

EX_ATTRIBUTE_PATTERN = re.compile(r'(width|height)="(\d+(?:\.\d+)?)ex"')
FONT_SIZE_PATTERN = re.compile(r'font-size="?(\d+(?:\.\d+)?)px"')
DEFAULT_FONT_SIZE_PX = 20.0


def collect_svg_files(paths: list[str]) -> list[Path]:
    """Gather SVG files from a list of file or directory paths."""
    svg_files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            svg_files.extend(sorted(path.rglob("*.svg")))
        elif path.is_file():
            svg_files.append(path)
        else:
            print(
                f"Warning: skipping {raw_path} (no such file or directory)",
                file=sys.stderr,
            )
    return svg_files


def convert_file(svg_file: Path) -> int:
    """
    Rewrite ex-based width/height attributes in one SVG file to px, in place.

    :param svg_file: path to the SVG file to fix
    :return: number of attributes rewritten
    """
    content = svg_file.read_text(encoding="utf-8")
    if not EX_ATTRIBUTE_PATTERN.search(content):
        return 0

    font_size_match = FONT_SIZE_PATTERN.search(content)
    font_size = (
        float(font_size_match.group(1)) if font_size_match else DEFAULT_FONT_SIZE_PX
    )
    ex_to_px = font_size / 2

    new_content, replaced = EX_ATTRIBUTE_PATTERN.subn(
        lambda match: f'{match.group(1)}="{float(match.group(2)) * ex_to_px:g}"',
        content,
    )
    if EX_ATTRIBUTE_PATTERN.search(new_content):
        raise RuntimeError(f"ex-based attributes remain in {svg_file}")

    svg_file.write_text(new_content, encoding="utf-8")
    return replaced


def main() -> None:
    """Process the given paths (or all SVGs next to this script) and report."""
    if sys.argv[1:]:
        svg_files = collect_svg_files(sys.argv[1:])
    else:
        svg_files = sorted(Path(__file__).resolve().parent.rglob("*.svg"))

    if not svg_files:
        print("No SVG files found.")
        return

    for svg_file in svg_files:
        replaced = convert_file(svg_file)
        status = f"{replaced} attribute(s) converted" if replaced else "nothing to do"
        print(f"{svg_file}: {status}")


if __name__ == "__main__":
    main()
