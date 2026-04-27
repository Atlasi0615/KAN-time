
from __future__ import annotations

import math
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont


def natural_key(path: Path):
    import re
    parts = re.split(r'(\d+)', path.name)
    return [int(p) if p.isdigit() else p for p in parts]


def collect_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    imgs.sort(key=natural_key)
    return imgs


def make_contact_sheet(
    folder: Path,
    out_path: Path,
    title: str,
    thumb_size=(280, 220),
    cols: int = 4,
    max_images: int | None = 16,
):
    images = collect_images(folder)
    if not images:
        raise FileNotFoundError(f"No images found in: {folder}")

    if max_images is not None:
        images = images[:max_images]

    rows = math.ceil(len(images) / cols)

    margin = 28
    gap_x = 20
    gap_y = 34
    title_h = 80
    label_h = 34

    canvas_w = margin * 2 + cols * thumb_size[0] + (cols - 1) * gap_x
    canvas_h = (
        title_h
        + margin
        + rows * (thumb_size[1] + label_h)
        + (rows - 1) * gap_y
        + margin
    )

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font_title = ImageFont.load_default()
    font_label = ImageFont.load_default()

    draw.text((margin, 20), title, fill="black", font=font_title)

    for idx, img_path in enumerate(images):
        row = idx // cols
        col = idx % cols

        x = margin + col * (thumb_size[0] + gap_x)
        y = title_h + row * (thumb_size[1] + label_h + gap_y)

        img = Image.open(img_path).convert("RGB")
        img.thumbnail(thumb_size)

        box = Image.new("RGB", thumb_size, (245, 245, 245))
        bx = (thumb_size[0] - img.width) // 2
        by = (thumb_size[1] - img.height) // 2
        box.paste(img, (bx, by))
        canvas.paste(box, (x, y))

        draw.rectangle(
            [x, y, x + thumb_size[0] - 1, y + thumb_size[1] - 1],
            outline=(180, 180, 180),
            width=1,
        )
        draw.text((x, y + thumb_size[1] + 8), img_path.name, fill="black", font=font_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved overview image to: {out_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create KAN plot overview contact sheets.")
    parser.add_argument("--kan-specific-dir", type=str, required=True,
                        help="Path like outputs_final/kan/<run>/kan_specific")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=16)
    args = parser.parse_args()

    base = Path(args.kan_specific_dir)

    jobs = [
        ("original_plot", "KAN overview: original"),
        ("sparse_plot", "KAN overview: sparsified"),
        ("pruned_plot", "KAN overview: pruned"),
    ]

    for subdir, title in jobs:
        folder = base / subdir
        if folder.exists():
            out_path = base / f"{subdir}_overview.png"
            make_contact_sheet(
                folder=folder,
                out_path=out_path,
                title=title,
                cols=args.cols,
                max_images=args.max_images,
            )
        else:
            print(f"Skip missing folder: {folder}")

    print("\nMarkdown snippet:")
    for subdir, _ in jobs:
        out_path = base / f"{subdir}_overview.png"
        if out_path.exists():
            print(f'![{subdir} overview]({out_path.as_posix()})')


if __name__ == "__main__":
    main()
