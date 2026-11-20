import numpy as np
from itertools import count
from typing import ClassVar, Any
from math import sqrt


class Spoke:
    """
    This class defines a single spoke as an object. It does so by storing the longitude and radius of each pixel of the spoke.
    """

    _id_counter: ClassVar = count(0)

    lon_arr: np.ndarray | None = None
    rad_arr: np.ndarray | None = None
    id: int | None = None
    track_id: str | None = None

    def __init__(
        self, lon_arr: np.ndarray, rad_arr: np.ndarray, id: int | None
    ) -> None:

        if lon_arr is None or rad_arr is None:
            raise Exception("Arguments cannot be None!")

        else:
            self.lon_arr = lon_arr
            self.rad_arr = rad_arr
            self.id = next(Spoke._id_counter)

    def set_track_id(self, id: int) -> None:
        self.track_id = f"t{id}"

    def get_track_id(self) -> str | None:
        return self.track_id

    def get_id(self) -> int | None:
        return self.id

    def get_lon_arr(self) -> np.ndarray:
        if self.lon_arr is None:
            raise Exception("lon_arr is None!")

        return self.lon_arr

    def get_rad_arr(self) -> np.ndarray:
        if self.rad_arr is None:
            raise Exception("rad_arr is None!")

        return self.rad_arr

    def get_centroid(self) -> tuple[float, float]:
        if self.lon_arr is None:
            raise Exception("lon_arr is None!")

        if self.lon_arr.size == 0:
            raise ValueError("Cannot find the centroid of an empty spoke")

        return (float(np.mean(self.lon_arr)), float(np.mean(self.rad_arr)))

    def get_distance(self, spoke: "Spoke") -> np.float32:
        """
        This function takes in a spoke and returns the distance between the self spoke
        and the argument spoke.
        It gets the centroid of each spoke, converts it into Cartesian coordiantes, and then calculates the L2 norm between the two points.
        """

        # NOTE: We have to convert the coordinates from polar to Cartesian, because otherwise the radius
        # value dominates over the longitude value. By converting to Cartesian coordinates, both dominate equally.

        # Get the centroid for both spokes
        self_lon, self_rad = self.get_centroid()
        spoke_lon, spoke_rad = spoke.get_centroid()

        # Convert to Cartesian coordinates
        self_radians, spoke_radians = np.deg2rad(self_lon), np.deg2rad(spoke_lon)
        self_x, self_y = (
            self_rad * np.cos(self_radians),
            self_rad * np.sin(self_radians),
        )
        spoke_x, spoke_y = (
            spoke_rad * np.cos(spoke_radians),
            spoke_rad * np.sin(spoke_radians),
        )

        a, b = abs(spoke_x - self_x), abs(spoke_y - self_y)

        return np.float32(sqrt(a**2 + b**2))

    @staticmethod
    def _coordinate_step(
        values: np.ndarray, groups: np.ndarray | None = None
    ) -> float | None:
        """Estimate raster spacing, optionally within individual radial rows."""
        if groups is None:
            groups = np.zeros(values.shape)

        spacings = []
        for group in np.unique(groups):
            unique_values = np.unique(values[groups == group])
            if unique_values.size < 2:
                continue
            differences = np.diff(unique_values)
            tolerance = max(1e-8, float(np.max(np.abs(unique_values))) * 1e-10)
            spacings.extend(differences[differences > tolerance].tolist())

        return min(spacings) if spacings else None

    @staticmethod
    def _shared_coordinate_step(
        first: np.ndarray,
        second: np.ndarray,
        first_groups: np.ndarray | None = None,
        second_groups: np.ndarray | None = None,
    ) -> float:
        steps = (
            Spoke._coordinate_step(first, first_groups),
            Spoke._coordinate_step(second, second_groups),
        )
        return max((step for step in steps if step is not None), default=1.0)

    @staticmethod
    def _centered_pixels(
        values: np.ndarray, centroid: float, step: float
    ) -> np.ndarray:
        """Convert centered coordinates to integer raster-cell indices."""
        return np.rint((values - centroid) / step).astype(np.int64)

    def get_iou(self, spoke: "Spoke") -> float:
        """Return centroid-aligned raster IoU between two spokes.

        Both coordinate sets are centered before being binned onto a shared
        grid. Nearby points therefore occupy the same pixel even when their
        floating-point coordinates are not exactly equal.
        """

        # NOTE: Think about replacing this function and instead comparing IoU scores by letting opencv
        # make a spoke into an image, and then comparing the images instead of lon and rad values.

        if self.get_lon_arr().size == 0 or spoke.get_lon_arr().size == 0:
            return 0.0

        curr_lon, curr_rad = self.get_centroid()
        spoke_lon, spoke_rad = spoke.get_centroid()

        curr_lon_arr = self.get_lon_arr()
        curr_rad_arr = self.get_rad_arr()
        s_lon_arr = spoke.get_lon_arr()
        s_rad_arr = spoke.get_rad_arr()

        lon_step = self._shared_coordinate_step(
            curr_lon_arr, s_lon_arr, curr_rad_arr, s_rad_arr
        )
        rad_step = self._shared_coordinate_step(curr_rad_arr, s_rad_arr)

        curr_pixels = zip(
            self._centered_pixels(curr_lon_arr, curr_lon, lon_step).tolist(),
            self._centered_pixels(curr_rad_arr, curr_rad, rad_step).tolist(),
        )
        spoke_pixels = zip(
            self._centered_pixels(s_lon_arr, spoke_lon, lon_step).tolist(),
            self._centered_pixels(s_rad_arr, spoke_rad, rad_step).tolist(),
        )

        curr_points = set(curr_pixels)
        spoke_points = set(spoke_pixels)
        intersection = len(curr_points & spoke_points)
        union = len(curr_points | spoke_points)

        return intersection / union if union else 0.0

    def get_area(self) -> int:
        assert self.lon_arr is not None
        return self.lon_arr.shape[0]

    def get_dict(self) -> dict[str, dict[str, Any]]:
        return {
            "spoke": {
                "id": self.id,
                "track_id": self.track_id,
                "lon_arr": self.get_lon_arr().tolist(),
                "rad_arr": self.get_rad_arr().tolist(),
            }
        }
