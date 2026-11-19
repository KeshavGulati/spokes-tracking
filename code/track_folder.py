"""Tracks spoke images taken consecutively in time.

Args:
    masks_path: Path to a folder containing masks
    rpjb_path: Path to a folder containing rpjb files corresponding to the masks

Returns:
    A JSON containing information about tracked spokes
    If visualize is specified, it returns two folders containing predicted spokes and observed_spokes with tracks.
"""

import colorsys
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils import load
from Frame import Frame
from Spoke import Spoke
from prediction import save_label_with_axes
from Tracker import Tracker


def add_to_map(frame_map: dict[str, Any], frame: Frame) -> dict[str, Any]:
    frame_map[frame.get_name()] = frame.get_dict()["frame"]
    return frame_map


def get_track_color(track_id: str) -> tuple[int, int, int]:
    track_number = int(track_id.removeprefix("t"))
    hue = ((track_number - 1) * 0.618033988749895) % 1.0
    return tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, 0.8, 1.0))


def _nearest_grid_indices(
    values: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(grid)
    sorted_grid = grid[order]
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(sorted_grid))))
    visible = (values >= sorted_grid[0] - tolerance) & (
        values <= sorted_grid[-1] + tolerance
    )
    clipped_values = np.clip(values, sorted_grid[0], sorted_grid[-1])

    right = np.clip(
        np.searchsorted(sorted_grid, clipped_values), 0, len(sorted_grid) - 1
    )
    left = np.maximum(right - 1, 0)
    nearest = np.where(
        np.abs(clipped_values - sorted_grid[left])
        <= np.abs(sorted_grid[right] - clipped_values),
        left,
        right,
    )
    return visible, order[nearest]


def render_frame_from_spokes(
    spokes: list[Spoke],
    name: str,
    lon_arr: np.ndarray,
    rad_arr: np.ndarray,
    output_dir: Path,
    suffix: str,
) -> Path:
    image = np.zeros((len(rad_arr), len(lon_arr), 3), dtype=np.uint8)

    for spoke in spokes:
        track_id = spoke.get_track_id()

        if track_id is None:
            raise ValueError("Cannot render a spoke without a track ID")

        visible_lons, cols = _nearest_grid_indices(spoke.get_lon_arr(), lon_arr)
        visible_rads, rows = _nearest_grid_indices(spoke.get_rad_arr(), rad_arr)
        visible = visible_lons & visible_rads
        image[rows[visible], cols[visible]] = get_track_color(track_id)

    return save_label_with_axes(
        image,
        list(lon_arr),
        list(rad_arr),
        str(output_dir),
        f"{Path(name).stem}_{suffix}",
    )


def temp_render(
    spokes: list[Spoke],
    lon_arr: np.ndarray,
    rad_arr: np.ndarray,
) -> np.ndarray:
    image = np.zeros((len(rad_arr), len(lon_arr), 3), dtype=np.uint8)

    for spoke in spokes:
        track_id = spoke.get_track_id()

        if track_id is None:
            raise ValueError("Cannot render a spoke without a track ID")

        visible_lons, cols = _nearest_grid_indices(spoke.get_lon_arr(), lon_arr)
        visible_rads, rows = _nearest_grid_indices(spoke.get_rad_arr(), rad_arr)
        visible = visible_lons & visible_rads
        image[rows[visible], cols[visible]] = get_track_color(track_id)

    return image


def temp_save(
    image: np.ndarray,
    name: str,
    output_dir: Path,
    suffix: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(name).stem}_{suffix}.png"
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if not cv2.imwrite(str(output_path), bgr_image):
        raise OSError(f"Could not save image to {output_path}")

    return output_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("masks_path")
    parser.add_argument("rpjb_path")
    parser.add_argument("-v", "--visualize", action="store_true")

    args = parser.parse_args()

    predictions_path: Path | None = None
    tracks_path: Path | None = None
    tracks_temp: Path | None = None
    output_path: Path = Path("./tracks.json")

    if args.visualize:
        predictions_path = Path("./predictions")
        tracks_path = Path("./tracks")

        tracks_temp = Path("./temp_tracks")

        predictions_path.mkdir(exist_ok=True)
        tracks_path.mkdir(exist_ok=True)

        # only for testing
        tracks_temp.mkdir(exist_ok=True)

    tracker: Tracker | None = None
    frame_map: dict[str, Any] = {}

    for mask, lon_arr, rad_arr, name in load(args.masks_path, args.rpjb_path):
        print(f"Processing {name}")
        frame = Frame(mask, name, lon_arr, rad_arr)
        predicted_spokes: list[Spoke] | None = None

        if tracker is None:
            tracker = Tracker(frame)
            predicted_spokes = tracker.get_last_spokes()
            observed_spokes = tracker.get_last_spokes()

        else:
            predicted_spokes, observed_spokes = tracker.track_spokes(frame)

        # Add the frame's info to a map to dump to JSON
        add_to_map(frame_map, frame)

        if predicted_spokes is None or observed_spokes is None:
            raise Exception("spokes cannot be None here!")

        if (
            args.visualize
            and predictions_path is not None
            and tracks_path is not None
            and tracks_temp is not None
        ):
            # save predictions
            render_frame_from_spokes(
                predicted_spokes,
                name,
                lon_arr,
                rad_arr,
                predictions_path,
                "predictions",
            )

            temp_save(
                temp_render(observed_spokes, lon_arr, rad_arr),
                name,
                tracks_temp,
                "temp_track",
            )

            # save observations with tracks
            render_frame_from_spokes(
                observed_spokes, name, lon_arr, rad_arr, tracks_path, "tracks"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(frame_map, indent=2) + "\n")
