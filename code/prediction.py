from argparse import ArgumentParser
from pathlib import Path
import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Saturn's gravitational parameter (km³/s²)
MU_SATURN = 3.7931208e7


def angular_velocity(r_km):
    """Keplerian angular velocity in radians/second."""
    return np.sqrt(MU_SATURN / r_km**3)


def predict_longitude(theta, r_km, dt_seconds):
    """Predict longitude in radians."""
    omega = angular_velocity(r_km)
    return (theta + omega * dt_seconds) % (2 * np.pi)


def save_label_with_axes(
    label_img: np.ndarray,
    lon_arr: list[int],
    rad_arr: list[int],
    save_dir: str,
    name: str,
    t=None,
):
    label_img = np.asarray(label_img)

    if t is not None:
        original_t = ""
        for s in name:
            if s.isdigit():
                original_t += s

        new_t = int(original_t) + t
        name = f"W{new_t}"

    # Squeeze single-channel images, but preserve three-channel RGB colors.
    if label_img.ndim == 3:
        if label_img.shape[2] == 1:
            label_img = label_img[:, :, 0]
        elif label_img.shape[2] != 3:
            raise ValueError("Images must be grayscale or have three RGB channels")

    label_img = label_img.astype(np.float32)

    label_img = label_img.astype(np.uint8)

    if label_img.shape[0] != len(rad_arr) or label_img.shape[1] != len(lon_arr):
        raise ValueError(
            f"Shape mismatch: label_img={label_img.shape}, "
            f"lon_arr={len(lon_arr)}, rad_arr={len(rad_arr)}"
        )

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    imshow_kwargs = {"cmap": "gray"} if label_img.ndim == 2 else {}
    ax.imshow(
        label_img,
        extent=(lon_arr[0], lon_arr[-1], np.min(rad_arr), np.max(rad_arr)),
        origin="upper",
        aspect="auto",
        **imshow_kwargs,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Radius")
    ax.set_title(name)

    fig.tight_layout()
    out_path = out_dir / f"{name}_with_axes.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path


def load(
    label_dir: str, rpjb_dir: str
) -> Generator[Tuple[np.ndarray, list[int], list[int], str], None, None]:
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

        yield (label_img, lon_arr, rad_arr, label_path.name)


def predict_next(
    label_img: np.ndarray,
    label_name: str,
    lon_arr: list[int],
    rad_arr: list[int],
    dt_seconds: int,
):
    """
    This function takes in a binary mask and its corresponding
    coordinates, and calculates the future position of all spokes
    in the mask, assuming that spokes are 1s in the original mask.
    """

    lon_arr = np.asarray(lon_arr).reshape(-1)
    rad_arr = np.asarray(rad_arr).reshape(-1)

    lon_order = np.argsort(lon_arr)
    sorted_lon = lon_arr[lon_order]

    rows, cols = np.nonzero(label_img)  # get all spoke pixels

    new_lon = np.rad2deg(
        predict_longitude(np.deg2rad(lon_arr[cols]), rad_arr[rows], dt_seconds)
    )

    # Convert physical longitudes to the nearest pixel column. Spokes that
    # leave the observed longitude range are no longer visible in the mask.
    visible = (new_lon >= sorted_lon[0]) & (new_lon <= sorted_lon[-1])
    rows, new_lon = rows[visible], new_lon[visible]
    right = np.clip(np.searchsorted(sorted_lon, new_lon), 0, len(sorted_lon) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(
        np.abs(new_lon - sorted_lon[left]) <= np.abs(sorted_lon[right] - new_lon),
        left,
        right,
    )

    predicted_img = np.zeros_like(label_img)
    predicted_img[rows, lon_order[nearest]] = 1

    return predicted_img


def map_future(label_dir: str, rpjb_dir: str, future_save_dir: str):
    prev_name, dt, new_img = None, 0, None
    prev_img = None

    for label_img, lon_arr, rad_arr, label_name in load(label_dir, rpjb_dir):
        if prev_name is None or prev_img is None:
            prev_name = label_name
            prev_img = label_img

        # TODO:

        if prev_name is not None:
            dt = get_time(label_name) - get_time(prev_name)

        new_img = predict_next(
            prev_img,
            label_name,
            lon_arr,
            rad_arr,
            dt,
        )

        if new_img is None:
            raise Exception()

        save_label_with_axes(
            new_img * 255, lon_arr, rad_arr, future_save_dir, label_name, dt
        )


def map_future_isolated(label_dir: str, rpjb_dir: str, save_dir: str):
    """
    This function takes in a mask and predicts the subsequent spoke locations based on the
    list of time it is given. It predicts the next mask from the previous mask, thereby accumulating
    any errors within the previous prediction.
    Finished using codex
    """

    frames = iter(load(label_dir, rpjb_dir))
    try:
        first_img, lon_arr, rad_arr, first_name = next(frames)
    except StopIteration:
        return

    # The first frame defines the grid for every prediction. Later frames are
    # used only for their timestamps, since their pixel dimensions may differ.
    save_label_with_axes(first_img * 255, lon_arr, rad_arr, save_dir, first_name)
    last_prediction = first_img
    prev_time = get_time(first_name)

    for _, _, _, label_name in frames:
        curr_time = get_time(label_name)
        dt = curr_time - prev_time
        predicted_img = predict_next(last_prediction, label_name, lon_arr, rad_arr, dt)

        save_label_with_axes(
            predicted_img * 255, lon_arr, rad_arr, save_dir, label_name
        )

        last_prediction = predicted_img
        prev_time = curr_time


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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("label_dir")
    parser.add_argument("rpjb_dir")
    parser.add_argument("save_dir")
    parser.add_argument("--isolated", action="store_true")
    args = parser.parse_args()

    if args.isolated:
        map_future_isolated(args.label_dir, args.rpjb_dir, args.save_dir)
    else:
        map_future(args.label_dir, args.rpjb_dir, args.save_dir)
