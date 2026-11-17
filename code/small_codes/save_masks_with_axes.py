from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import re
import cv2
import scipy.io

def mask_paths(label_dir):
    paths = sorted(p for p in Path(label_dir).iterdir()
                   if p.is_file() and p.suffix.lower() == ".png")
    if not paths:
        raise ValueError(f"No PNG masks found in {label_dir}")
    return paths


def timestamp(path):
    match = re.match(r"W\d+", Path(path).stem)
    if match is None:
        raise ValueError(f"Expected a filename starting with W and a timestamp: {path}")
    return match.group()


def get_label_coord_pair(label_dir: str, rpjb_dir: str):
    """Yield masks and axes, matching PNG and RPJB files by timestamp."""
    coordinates = {}
    for path in sorted(Path(rpjb_dir).iterdir()):
        if path.is_file() and path.suffix.lower() == ".rpjb":
            key = timestamp(path)
            if key in coordinates:
                raise ValueError(f"Multiple RPJB files for {key}")
            coordinates[key] = path

    for label in mask_paths(label_dir):
        key = timestamp(label)
        if key not in coordinates:
            raise ValueError(f"No matching RPJB file for {label.name}")
        label_img = cv2.imread(str(label), cv2.IMREAD_GRAYSCALE)
        if label_img is None:
            raise ValueError(f"Cannot read mask: {label}")
        idl = scipy.io.readsav(coordinates[key])
        min_lon, max_lon, min_rad, max_rad = (
            np.asarray(idl[field]).item()
            for field in ("mnlon", "mxlon", "mnrad", "mxrad")
        )
        lon_arr = np.linspace(min_lon, max_lon, label_img.shape[1])
        rad_arr = np.linspace(min_rad, max_rad, label_img.shape[0])
        yield label_img, lon_arr, rad_arr



def save_label_with_axes(label_img, lon_arr, rad_arr, save_dir, name, t=None):
    label_img = np.asarray(label_img)
    name = Path(name).stem

    if (t is not None):
        original_t = ""
        for s in name:
            if (s.isdigit()):
                original_t += s

        new_t = int(original_t) + t
        name = f"W{new_t}"


    # Make it 2D grayscale
    if label_img.ndim == 3:
        if label_img.shape[2] == 1:
            label_img = label_img[:, :, 0]
        elif label_img.shape[2] == 3:
            # if it is RGB, convert to grayscale
            label_img = np.mean(label_img, axis=2)
        else:
            label_img = label_img[:, :, 0]

    label_img = label_img.astype(np.float32)
    if label_img.size and label_img.max() <= 1:
        label_img = label_img * 255
    label_img = np.clip(label_img, 0, 255).astype(np.uint8)

    if label_img.shape[0] != len(rad_arr) or label_img.shape[1] != len(lon_arr):
        raise ValueError(
            f"Shape mismatch: label_img={label_img.shape}, "
            f"lon_arr={len(lon_arr)}, rad_arr={len(rad_arr)}"
        )

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(
        label_img,
        cmap="gray",
        vmin=0, vmax=255,
        extent=[lon_arr[0], lon_arr[-1], rad_arr[0], rad_arr[-1]],
        aspect="auto",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Radius")
    ax.set_title(name)

    fig.tight_layout()
    out_path = out_dir / f"{name}_with_axes.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('label_dir')
    parser.add_argument('rpjb_dir')
    parser.add_argument('save_dir')
    args = parser.parse_args()


    try:
        paths = mask_paths(args.label_dir)
        pairs = get_label_coord_pair(args.label_dir, args.rpjb_dir)
        for path, (label_img, lon_arr, rad_arr) in zip(paths, pairs):
            save_label_with_axes(label_img, lon_arr, rad_arr, args.save_dir, path.stem)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved {len(paths)} masks with axes to {args.save_dir}")
