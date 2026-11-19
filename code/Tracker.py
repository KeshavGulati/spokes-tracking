import numpy as np
from Frame import *
from itertools import count
from utils import predict_longitude


class TrackerUninitialized(Exception):
    """
    Raised when some operation is being attempted on a tracker but the tracker is uninitialized.
    """


class LocalCounter:
    curr: int = 0

    def __init__(self) -> None:
        self.curr = -1

    def next(self) -> int:
        self.curr -= 1
        return self.curr + 1

    def reset(self) -> None:
        self.curr = -1


class Tracker:
    last_frame: Frame | None = None
    last_spokes: list[Spoke]
    seen_spokes: list[Spoke]
    save_frames: bool
    _id_counter = count(1)

    # NOTE: We cannot store the predicted frame because we don't know dt at the time of initialization.

    def __init__(self, frame: Frame, save_frames=False) -> None:
        self.last_frame = frame
        self.last_spokes = frame.collect_spokes()
        self.seen_spokes = []
        self.save_frames = save_frames

        # For the first frame, assign each spoke a track ID.
        for spoke in self.last_spokes:
            spoke.set_track_id(next(self._id_counter))
            self.seen_spokes.append(spoke)

    def get_last_spokes(self) -> list[Spoke]:
        return self.last_spokes

    @staticmethod
    def _match_spokes(
        first_spokes: list[Spoke], second_spokes: list[Spoke], lon_arr: np.ndarray
    ) -> list[tuple[Spoke, Spoke]]:
        """Greedily find one-to-one matches using distance and shape similarity."""
        candidates = []
        for first_index, first_spoke in enumerate(first_spokes):
            for second_index, second_spoke in enumerate(second_spokes):
                distance = first_spoke.get_distance(second_spoke)
                iou = first_spoke.get_iou(second_spoke)

                distance_scale = lon_arr.shape[
                    0
                ]  # divide by the scale to make the highest possible distance = 1

                # the lower the cost, the higher the similarity
                # Since IoU can change over time even for the same spoke (spoke evolution
                # or new spoke enters), I want the distance to dominate the cost.
                # Greater the distance, the more the cost, and greater the IoU the less the cost.
                cost = float(distance / distance_scale) - (10 * iou) ** 2
                candidates.append((cost, first_index, second_index))

        matches = []
        used_first = set()
        used_second = set()
        for _, first_index, second_index in sorted(candidates):
            if first_index in used_first or second_index in used_second:
                continue

            matches.append((first_spokes[first_index], second_spokes[second_index]))
            used_first.add(first_index)
            used_second.add(second_index)

        return matches

    def track_spokes(self, observed_frame: Frame) -> tuple[list[Spoke], list[Spoke]]:
        if self.last_frame is None:
            raise TrackerUninitialized()

        dt = observed_frame.time - self.last_frame.time

        observed_spokes = observed_frame.collect_spokes()

        # Predict the current frame's spokes from the previous frame's spokes
        predicted_spokes: list[Spoke] = []

        for last_spoke in self.last_spokes:
            last_spoke_lon_arr = last_spoke.get_lon_arr()
            last_spoke_rad_arr = last_spoke.get_rad_arr()
            last_spoke_id = last_spoke.get_id()
            new_lon = []

            # predict each new longitude value from the previous spoke, and assign the new spoke the same id.
            for lon, rad in zip(last_spoke_lon_arr, last_spoke_rad_arr):
                new_lon.append(np.rad2deg(predict_longitude(np.deg2rad(lon), rad, dt)))

            track_id = last_spoke.get_track_id()

            if track_id is None:
                raise Exception("track_id cannot be None here!")

            track_id = int(track_id[1:])

            new_spoke = Spoke(np.array(new_lon), last_spoke_rad_arr, last_spoke_id)
            new_spoke.set_track_id(track_id)
            predicted_spokes.append(new_spoke)

        matched_observed = set()
        for predicted_spoke, observed_spoke in self._match_spokes(
            predicted_spokes, observed_spokes, observed_frame.get_lon_arr()
        ):
            if predicted_spoke.track_id is None:
                continue

            observed_spoke.track_id = predicted_spoke.track_id
            matched_observed.add(id(observed_spoke))

        # Observations without a matching prediction are new tracks. Predictions
        # without an observation are allowed to exit without creating a new track.
        for observed_spoke in observed_spokes:
            if id(observed_spoke) not in matched_observed:
                observed_spoke.set_track_id(next(self._id_counter))

        self.last_frame = observed_frame
        self.last_spokes = observed_spokes
        self.seen_spokes.extend(observed_spokes)

        # return predicted_spokes
        return predicted_spokes, observed_spokes
