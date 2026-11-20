"""Generate compact, inspectable transitions from the real spoke tracker.

Run from the repository root:
    venv/bin/python demo/build_demo_data.py

The browser demo only reads the resulting JSON. It does not reimplement the
motion model, scoring, or greedy assignment algorithm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from Frame import Frame  # noqa: E402
from Spoke import Spoke  # noqa: E402
from Tracker import Tracker  # noqa: E402
from track_folder import (  # noqa: E402
    _nearest_grid_indices,
    get_track_color,
)
from utils import load  # noqa: E402


def spoke_dict(
    spoke: Spoke, lon_arr: np.ndarray, rad_arr: np.ndarray
) -> dict[str, Any]:
    """Render points exactly as track_folder does, compacted to flat pixels."""
    visible_lons, cols = _nearest_grid_indices(spoke.get_lon_arr(), lon_arr)
    visible_rads, rows = _nearest_grid_indices(spoke.get_rad_arr(), rad_arr)
    visible = visible_lons & visible_rads
    rows = rows[visible]
    cols = cols[visible]
    flat_pixels = sorted(set((rows * len(lon_arr) + cols).tolist()))
    lon, rad = spoke.get_centroid()
    track_id = spoke.get_track_id()

    return {
        "component_id": spoke.get_id(),
        "track_id": track_id,
        "color": get_track_color(track_id) if track_id else [255, 255, 255],
        "area": spoke.get_area(),
        "lon": round(lon, 4),
        "rad": round(rad, 2),
        "label_xy": [round(float(np.mean(cols)), 2), round(float(np.mean(rows)), 2)],
        "points": flat_pixels,
    }


def build(masks_dir: Path, rpjb_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker: Tracker | None = None
    original_matcher = Tracker._match_spokes
    transitions: list[dict[str, Any]] = []

    for mask, lon_arr, rad_arr, name in load(str(masks_dir), str(rpjb_dir)):
        frame = Frame(mask, name, lon_arr, rad_arr)
        if tracker is None:
            tracker = Tracker(frame)
            continue

        previous_spokes = tracker.get_last_spokes()
        previous_name = tracker.last_frame.get_name() if tracker.last_frame else ""
        previous_time = tracker.last_frame.time if tracker.last_frame else frame.time
        captured: dict[str, Any] = {}

        def capture_matches(
            predicted: list[Spoke],
            observed: list[Spoke],
            current_lons: np.ndarray,
            current_rads: np.ndarray,
        ) -> list[tuple[Spoke, Spoke]]:
            # Use the production matcher for the decisions. The additional
            # values expose the exact comparisons made by that matcher.
            matches = original_matcher(predicted, observed, current_lons, current_rads)
            tll, tlr = current_lons[0], current_rads[0]
            brl, brr = current_lons[-1], current_rads[-1]
            tlx, tly = tlr * np.sin(tll), tlr * np.cos(tll)
            brx, bry = brr * np.sin(brl), brr * np.cos(brl)
            distance_scale = np.float32(np.hypot(brx - tlx, bry - tly))
            candidates = []

            for prediction_index, prediction in enumerate(predicted):
                for observation_index, observation in enumerate(observed):
                    distance = float(prediction.get_distance(observation))
                    shape_similarity = prediction.get_iou(observation)
                    normalized_distance = distance / float(distance_scale)
                    cost = normalized_distance - (10 * shape_similarity) ** 2
                    candidates.append(
                        {
                            "prediction": prediction_index,
                            "observation": observation_index,
                            "cost": cost,
                            "distance": normalized_distance,
                            "shape": shape_similarity,
                        }
                    )

            candidates.sort(key=lambda c: (c["cost"], c["prediction"], c["observation"]))
            captured["candidates"] = candidates
            captured["matches"] = [
                (prediction.get_id(), observation.get_id())
                for prediction, observation in matches
            ]
            return matches

        tracker._match_spokes = capture_matches  # type: ignore[method-assign]
        predicted, observed = tracker.track_spokes(frame)
        transition_index = len(transitions)
        candidate_copy = captured["candidates"]

        for rank, candidate in enumerate(candidate_copy, start=1):
            candidate["rank"] = rank
            candidate["cost"] = round(candidate["cost"], 8)
            candidate["distance"] = round(candidate["distance"], 8)
            candidate["shape"] = round(candidate["shape"], 8)

        prediction_component_to_index = {spoke.get_id(): i for i, spoke in enumerate(predicted)}
        observation_component_to_index = {spoke.get_id(): i for i, spoke in enumerate(observed)}
        accepted_pairs = [
            [
                prediction_component_to_index[prediction_id],
                observation_component_to_index[observation_id],
            ]
            for prediction_id, observation_id in captured["matches"]
        ]

        record = {
            "index": transition_index,
            "previous_name": previous_name,
            "current_name": name,
            "previous_time": previous_time,
            "current_time": frame.time,
            "dt_seconds": frame.time - previous_time,
            "width": len(lon_arr),
            "height": len(rad_arr),
            "lon_range": [round(float(lon_arr[0]), 3), round(float(lon_arr[-1]), 3)],
            "rad_range": [round(float(rad_arr[0]), 2), round(float(rad_arr[-1]), 2)],
            "previous": [spoke_dict(s, lon_arr, rad_arr) for s in previous_spokes],
            "predictions": [spoke_dict(s, lon_arr, rad_arr) for s in predicted],
            "observations": [spoke_dict(s, lon_arr, rad_arr) for s in observed],
            "candidates": candidate_copy,
            "accepted_pairs": accepted_pairs,
        }
        file_name = f"transition_{transition_index:02d}.json"
        (output_dir / file_name).write_text(json.dumps(record, separators=(",", ":")) + "\n")

        observation_ids = {spoke.get_track_id() for spoke in observed}
        previous_ids = {spoke.get_track_id() for spoke in previous_spokes}
        new_ids = sorted(observation_ids - previous_ids)
        ended_ids = sorted(previous_ids - observation_ids)
        transitions.append(
            {
                "index": transition_index,
                "file": file_name,
                "previous_frame": transition_index + 1,
                "current_frame": transition_index + 2,
                "dt_seconds": record["dt_seconds"],
                "predictions": len(predicted),
                "observations": len(observed),
                "candidate_count": len(candidate_copy),
                "matches": len(accepted_pairs),
                "new_tracks": len(new_ids),
                "ended_tracks": len(ended_ids),
            }
        )
        print(
            f"Frames {transition_index + 1:02d} → {transition_index + 2:02d}: "
            f"{len(predicted)} predictions, {len(observed)} observations, "
            f"{len(candidate_copy)} comparisons"
        )

    metadata = {"transitions": transitions}
    (output_dir / "index.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(transitions)} transitions to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks", type=Path, default=ROOT / "data" / "masks")
    parser.add_argument("--rpjb", type=Path, default=ROOT / "data" / "rpjb")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    build(args.masks, args.rpjb, args.output)


if __name__ == "__main__":
    main()
