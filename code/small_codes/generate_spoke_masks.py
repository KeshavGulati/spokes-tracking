import glob
import io
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_FILE = SCRIPT_DIR / "../../data/prompt.txt"
REFERENCES_DIR = SCRIPT_DIR / "../../data/masks"
OUTPUT_DIR = SCRIPT_DIR / "synthetic"
# MODEL = "gemini-3-pro-image-preview"  # or "gemini-3.1-flash-image"; verify with client.models.list()
MODEL = "gemini-3.1-flash-lite-image"
PROMPT = PROMPT_FILE.read_text()
refs = sorted(glob.glob(str(REFERENCES_DIR / "*.png")))
OUT_SIZE = (1488, 380)  # (w, h)


def generate(i, n_refs=8):
    if not 0 < n_refs <= len(refs):
        raise ValueError(f"n_refs must be between 1 and {len(refs)}")
    start = (i * n_refs) % len(refs)
    ref_paths = [refs[(start + j) % len(refs)] for j in range(n_refs)]
    ref_imgs = [Image.open(p).convert("L") for p in ref_paths]
    client = genai.Client()  # reads GEMINI_API_KEY
    resp = client.models.generate_content(
        model=MODEL,
        contents=[PROMPT, *ref_imgs],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )
    for candidate in resp.candidates or []:
        for part in candidate.content.parts if candidate.content else []:
            if part.inline_data:
                img = Image.open(io.BytesIO(part.inline_data.data)).convert("L")
                img = img.resize(OUT_SIZE, Image.Resampling.LANCZOS)
                arr = np.array(img)
                _, mask = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)
                if is_valid(mask):
                    output_file = OUTPUT_DIR / f"mask_{i:05d}.png"
                    if not cv2.imwrite(str(output_file), mask):
                        raise OSError(f"Could not write mask to {output_file}")
                    return True
    return False


def is_valid(mask):
    n, _ = cv2.connectedComponents(mask)
    n_regions = n - 1
    touches_border = (
        mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
    )
    return 5 <= n_regions <= 12 and not touches_border


def main():
    limit = int(input("How many images do you want to generate? "))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    made = 0
    while made < limit:
        if generate(made):
            made += 1


if __name__ == "__main__":
    main()
