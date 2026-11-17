import cv2
import numpy as np
import scipy.io
from pathlib import Path
from typing import Generator, Tuple

MU_SATURN = 3.7931208e7


def angular_velocity(r_km):
    """Keplerian angular velocity in radians/second."""
    return np.sqrt(MU_SATURN / r_km**3)


def predict_longitude(theta, r_km, dt_seconds):
    """Predict longitude in radians."""
    omega = angular_velocity(r_km)
    return (theta + omega * dt_seconds) % (2 * np.pi)


def get_time(label_path: str):
    """
    This function takes in a mask and returns the time when that image was taken. It assumes
    that the number in the mask name is represents that time relative to some event.
    Effectively, it strips out all characters from the mask name and returns just the digits.
    """

    name = Path(label_path).name
    digits = ""
    for character in name[1:]:  # The first character is W.
        if not character.isdigit():
            break
        digits += character
    return int(digits)


def load(
    label_dir: str, rpjb_dir: str
) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray, str], None, None]:
    """
    This is a generator function that yields pairs of a mask and its corresponding
    radius and longitude array derived from the appropriate rpjb file.
    """

    coordinates = {}
    for path in Path(rpjb_dir).glob("*.rpjb"):
        timestamp = get_time(path.name)
        if timestamp in coordinates:
            raise ValueError(f"Multiple RPJB files for timestamp {timestamp}")
        coordinates[timestamp] = path

    for label_path in sorted(
        Path(label_dir).glob("*.png"), key=lambda p: get_time(p.name)
    ):
        timestamp = get_time(label_path.name)
        if timestamp not in coordinates:
            raise ValueError(f"No matching RPJB file for {label_path.name}")
        rpjb_path = coordinates[timestamp]
        # load the label
        label = Path(label_path)
        label_img = cv2.imread(label, cv2.IMREAD_GRAYSCALE)

        if label_img is None:
            raise Exception()

        label_img = label_img.astype(np.float32) / 255.0

        # load the rpjb
        rpjb = Path(rpjb_path)
        idl = scipy.io.readsav(rpjb)
        min_lon, max_lon = idl["mnlon"], idl["mxlon"]
        min_rad, max_rad = idl["mnrad"], idl["mxrad"]

        lon_arr = np.linspace(min_lon, max_lon, label_img.shape[1])
        rad_arr = np.linspace(
            max_rad, min_rad, label_img.shape[0]
        )  # the rad_arr goes from max to min because we save the images considering that Saturn is located towards the bottom of the image.

        temp = np.zeros((len(rad_arr), len(lon_arr)))

        if label_img.shape != temp.shape:
            print("label.shape != temp.shape")
            raise Exception()

        yield (label_img, np.array(lon_arr), np.array(rad_arr), label_path.name)
