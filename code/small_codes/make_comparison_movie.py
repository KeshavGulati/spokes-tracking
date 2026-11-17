"""Render two filename-sorted PNG sequences as a single 1x2 movie."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from make_movie import make_movie


def png_paths(directory):
    paths = sorted(p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() == ".png")
    if not paths:
        raise ValueError(f"No PNG images found in {directory}")
    return paths


def comparison_frames(left, right, panel_width=960, panel_height=720):
    if len(left) != len(right):
        raise ValueError(f"Frame counts must match: left={len(left)}, right={len(right)}")
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    frames = []
    # Normalize each sequence to its first frame, just as make_movie.py does.
    with Image.open(left[0]) as image:
        left_size = image.size
    with Image.open(right[0]) as image:
        right_size = image.size
    for left_path, right_path in zip(left, right):
        canvas = Image.new("RGB", (2 * panel_width, panel_height + 64), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (path, size, title) in enumerate([
            (left_path, left_size, "Original"),
            (right_path, right_size, "Prediction"),
        ]):
            with Image.open(path) as source:
                image = source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
            image = ImageOps.contain(image, (panel_width, panel_height), Image.Resampling.LANCZOS)
            x = index * panel_width
            canvas.paste(image, (x + (panel_width - image.width) // 2,
                                 64 + (panel_height - image.height) // 2))
            draw.text((x + panel_width // 2, 8), title, fill="black", font=font, anchor="mt")
            draw.text((x + panel_width // 2, 34), path.name, fill="black", font=font, anchor="mt")
        draw.line((panel_width, 0, panel_width, canvas.height), fill="gray")
        frames.append(np.asarray(canvas)[:, :, ::-1])
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left_dir", nargs="?", type=Path, default=Path("081_SPKMVLFLP/png"))
    parser.add_argument("right_dir", nargs="?", type=Path, default=Path("predictions1"))
    parser.add_argument("--fps", type=float, default=24)
    parser.add_argument("-o", "--output", type=Path, default=Path("original_vs_predictions.mp4"))
    args = parser.parse_args()
    try:
        frames = comparison_frames(png_paths(args.left_dir), png_paths(args.right_dir))
        make_movie(frames, fps=args.fps, output_path=args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Saved {len(frames)} paired frames at {args.fps:g} fps to {args.output}")


if __name__ == "__main__":
    main()
