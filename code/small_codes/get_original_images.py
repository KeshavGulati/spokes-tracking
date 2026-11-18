from pathlib import Path
import re
import requests
from argparse import ArgumentParser

BASE_URL = "https://opus.pds-rings.seti.org/opus/api"
OUTPUT_DIR = Path("cassini_images")
OUTPUT_DIR.mkdir(exist_ok=True)

session = requests.Session()


def filename_to_opus_id(filename):
    name = Path(filename.strip()).name.upper()

    match = re.fullmatch(r"([NW]\d{10})(?:_\d+)?(?:\.IMG)?", name)

    if not match:
        raise ValueError(f"Invalid Cassini ISS filename: {filename}")

    return f"co-iss-{match.group(1).lower()}"


def download_image(filename):
    opus_id = filename_to_opus_id(filename)

    response = session.get(
        f"{BASE_URL}/files/{opus_id}.json",
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()["data"][opus_id]
    preview_files = data.get("browse_full", [])

    image_urls = [url for url in preview_files if url.upper().endswith(".PNG")]

    if not image_urls:
        raise FileNotFoundError(f"No preview image found: {opus_id}")

    for url in image_urls:
        output_path = OUTPUT_DIR / url.rsplit("/", 1)[-1]

        if output_path.exists():
            print(f"Skipping {output_path.name}")
            continue

        with session.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with output_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        print(f"Downloaded {output_path.name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filename")

    args = parser.parse_args()
    filename = args.filename

    with open(filename) as f:
        filenames = [line.strip() for line in f if line.strip()]

    for filename in filenames:
        try:
            download_image(filename)
        except (
            ValueError,
            requests.RequestException,
            FileNotFoundError,
            KeyError,
        ) as e:
            print(f"Failed {filename}: {e}")
