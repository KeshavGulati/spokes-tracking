import numpy as np
from collections import deque
from Spoke import *
from utils import predict_longitude, get_time


class Frame:
    """
    This class represent a single binary mask using its intensity values, and also stores the corresponding
    longitude and radius values for each point in the mask.
    """

    mask: np.ndarray
    name: str
    time: int
    lon_arr: np.ndarray
    rad_arr: np.ndarray

    def __init__(
        self, mat: np.ndarray, name: str, lon_arr: np.ndarray, rad_arr: np.ndarray
    ) -> None:

        # ensure that the mask is two dimesional
        if len(mat.shape) != 2:
            raise Exception("The frame can only be two dimensional!")

        self.mask = mat
        self.name = name
        self.time = get_time(name)
        self.lon_arr = lon_arr
        self.rad_arr = rad_arr
        self.spokes: list[Spoke] | None = None

    def get_name(self) -> str:
        return self.name

    def get_time(self):
        return self.time

    def get_lon_arr(self):
        return self.lon_arr

    def get_rad_arr(self):
        return self.rad_arr

    @classmethod
    def create_frame(
        cls, mat: np.ndarray, name: str, lon_arr: np.ndarray, rad_arr: np.ndarray
    ) -> "Frame":
        return cls(mat, name, lon_arr, rad_arr)

    def _collect_spokes_helper(
        self, y: int, x: int, visited: set[tuple[int, int]]
    ) -> tuple[Spoke, set[tuple[int, int]]]:
        """
        This is a private helper function for the collect_spokes function. It runs a bfs on a given lon, rad coordinate
        and searches for neighboring coordinates in the mask that are connected to the given coordinate and have
        the intensity = 1.
        @param y: The radius value of some point
        x: The longitude value of some point
        visited: The set of all visited coordinates

        @return Returns a Spoke instance and a visited set.
        """

        queue = deque([(y, x)])
        visited.add((y, x))
        spoke_lons = []
        spoke_rads = []

        while queue:
            curr_y, curr_x = queue.popleft()
            spoke_lons.append(self.lon_arr[curr_x])
            spoke_rads.append(self.rad_arr[curr_y])

            neighbors = (
                (curr_y - 1, curr_x),
                (curr_y + 1, curr_x),
                (curr_y, curr_x - 1),
                (curr_y, curr_x + 1),
            )
            for ny, nx in neighbors:
                if (
                    0 <= ny < len(self.rad_arr)
                    and 0 <= nx < len(self.lon_arr)
                    and (ny, nx) not in visited
                    and self.mask[ny, nx] == 1
                ):
                    visited.add((ny, nx))
                    queue.append((ny, nx))

        return Spoke(np.array(spoke_lons), np.array(spoke_rads), None), visited

    def collect_spokes(self) -> list[Spoke]:
        """
        This method collects all spokes in a given mask.
        @param None
        @return A list of Spoke objects
        """

        if self.spokes is not None:
            return self.spokes

        spoke_list: list[Spoke] = []
        visited: set[tuple[int, int]] = set()

        if self.mask is None:
            raise Exception("Mask is None!")

        for y in range(self.mask.shape[0]):
            for x in range(self.mask.shape[1]):
                if self.mask[y, x] == 1 and (y, x) not in visited:
                    new_spoke, visited = self._collect_spokes_helper(y, x, visited)
                    spoke_list.append(new_spoke)

        self.spokes = spoke_list
        return spoke_list

    def _render_frame_from_spokes(
        self,
        spokes: list[Spoke],
        name: str,
        lon_arr: np.ndarray,
        rad_arr: np.ndarray,
    ) -> "Frame":
        """
        This function takes in a list of spokes, lon_arr, and rad_arr and constructs and returns a Frame object from it.
        """

        predicted_img = np.zeros((len(rad_arr), len(lon_arr)), dtype=self.mask.dtype)
        lon_order = np.argsort(lon_arr)
        sorted_lon = lon_arr[lon_order]
        rad_order = np.argsort(rad_arr)
        sorted_rad = rad_arr[rad_order]
        lon_tolerance = 1e-12 * max(1.0, float(np.max(np.abs(sorted_lon))))
        rad_tolerance = 1e-12 * max(1.0, float(np.max(np.abs(sorted_rad))))

        for spoke in spokes:
            spoke_lons = spoke.get_lon_arr()
            spoke_rads = spoke.get_rad_arr()

            # Discard points outside the target coordinate grid.
            visible = (spoke_lons >= sorted_lon[0] - lon_tolerance) & (
                spoke_lons <= sorted_lon[-1] + lon_tolerance
            )
            visible &= (spoke_rads >= sorted_rad[0] - rad_tolerance) & (
                spoke_rads <= sorted_rad[-1] + rad_tolerance
            )
            visible_lons = spoke_lons[visible]
            visible_rads = spoke_rads[visible]

            lon_right = np.clip(
                np.searchsorted(sorted_lon, visible_lons), 0, len(sorted_lon) - 1
            )
            lon_left = np.maximum(lon_right - 1, 0)
            nearest_lon = np.where(
                np.abs(visible_lons - sorted_lon[lon_left])
                <= np.abs(sorted_lon[lon_right] - visible_lons),
                lon_left,
                lon_right,
            )

            rad_right = np.clip(
                np.searchsorted(sorted_rad, visible_rads), 0, len(sorted_rad) - 1
            )
            rad_left = np.maximum(rad_right - 1, 0)
            nearest_rad = np.where(
                np.abs(visible_rads - sorted_rad[rad_left])
                <= np.abs(sorted_rad[rad_right] - visible_rads),
                rad_left,
                rad_right,
            )

            rows = rad_order[nearest_rad]
            cols = lon_order[nearest_lon]
            predicted_img[rows, cols] = 1

        return self.create_frame(predicted_img, name, lon_arr, rad_arr)

    def _predict_next_spoke(self, spoke: Spoke, dt: int) -> Spoke:
        """
        This function predicts the next spoke from the given spoke assuming Keplerian motion.
        """

        lon_arr, rad_arr = spoke.get_lon_arr(), spoke.get_rad_arr()
        new_lon = np.rad2deg(predict_longitude(np.deg2rad(lon_arr), rad_arr, dt))
        track_id = spoke.get_track_id()
        if track_id is None:
            raise Exception("Track_id should not be None!")

        track_id = int(track_id[1:])

        new_spoke = Spoke(new_lon, rad_arr, spoke.get_id())
        new_spoke.set_track_id(track_id)

        return new_spoke

    def predict_next(
        self,
        dt_seconds: int,
        name: str,
        lon_arr: np.ndarray,
        rad_arr: np.ndarray,
    ) -> Frame:
        """
        This function predicts the next frame based on the current frame assuming simple Keplerian motion.
        """

        observed_spokes = self.collect_spokes()
        predicted_spokes = []

        for spoke in observed_spokes:
            new_spoke = self._predict_next_spoke(spoke, dt_seconds)
            predicted_spokes.append(new_spoke)

        new_frame = self._render_frame_from_spokes(
            predicted_spokes, name, lon_arr, rad_arr
        )

        return new_frame

    def get_dict(self):
        return {"frame": [spoke.get_dict() for spoke in self.collect_spokes()]}
