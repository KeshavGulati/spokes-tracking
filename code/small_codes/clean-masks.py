from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import cv2


def _bfs(mask: np.ndarray, start: tuple[int, int]) -> tuple[int, set[tuple[int, int]]]:
    queue: list[tuple[int | None, int | None]] = [start]
    queued: set[tuple[int, int]] = {start}
    local_visited: set[tuple[int, int]] = set()
    count = 0
    w, h = mask.shape

    while len(queue) > 0:
        x, y = queue.pop(0)
        if x is None:
            continue

        assert y is not None

        neighbors = [
            (a, b)
            for a in [x - 1, x, x + 1]
            for b in [y - 1, y, y + 1]
            if (a, b) != (x, y) and -1 < a < w and -1 < b < h
        ]

        for nx, ny in neighbors:
            if nx is None or ny is None or (nx, ny) in queued:
                continue

            elif mask[nx, ny] == 1:
                queue.append((nx, ny))
                queued.add((nx, ny))

        local_visited.add((x, y))
        count += 1

    return count, local_visited


def clean_mask(mask: np.ndarray, threshold: int) -> np.ndarray:

    visited: set[tuple[int, int]] = set()

    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if (row, col) in visited:
                continue

            p = mask[row, col]

            if p == 1:
                count, local_visited = _bfs(mask, (row, col))

                if count < threshold:
                    for x, y in local_visited:
                        mask[x, y] = 0

                visited |= local_visited

    return mask


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("masks_path", help="The path to the folder containing masks")
    parser.add_argument(
        "--threshold",
        default=3,
        type=int,
        help="The area of threshold to consider a spoke",
    )
    args = parser.parse_args()

    mask_dir = Path(args.masks_path)
    output_dir = Path("./clean_masks")
    output_dir.mkdir(parents=True, exist_ok=True)

    for mask_file in mask_dir.iterdir():
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError(f"mask file = {mask_file.name}")

        mask = mask.astype(np.float32) / 255.0
        new_mask = clean_mask(mask, args.threshold)
        output_path = output_dir / f"{mask_file.with_suffix('.png').name}"

        Image.fromarray((new_mask * 255).astype(np.uint8)).save(output_path)
        print(f"Cleaned and saved {mask_file.name}")
