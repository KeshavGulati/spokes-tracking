import json
import argparse
import math
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import tempfile
import subprocess
import base64
from IPython.display import HTML, display
import uuid

"""
    This script takes in a directory of .png files, sorts them by their name (assuming that the file name corresponds to the time that the image was taken), and then makes a movie out of it.
"""

def make_movie(frames:np.ndarray, labels=None, fps:float=24, output_path=None):
    # code written by Claude
    """
    frames: list of numpy arrays (H, W, 3), BGR format (as from cv2/detector output)
    fps: playback frame rate
    """
    if len(frames) == 0:
        raise ValueError("frames list is empty")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")
    if labels is not None and len(labels) != len(frames):
        raise ValueError("number of labels must match number of frames")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg must be installed and available on PATH")

    height, width = frames[0].shape[:2]
    with tempfile.TemporaryDirectory() as tmp_dir:
        for index, frame in enumerate(frames):
            frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                raise ValueError("frames must contain uint8 pixels")
            if frame.ndim == 2:
                image = Image.fromarray(frame).convert("RGB")
            elif frame.ndim == 3 and frame.shape[2] in (3, 4):
                # The function accepts BGR/BGRA arrays, as documented.
                image = Image.fromarray(frame[:, :, [2, 1, 0]])
            else:
                raise ValueError("frames must be grayscale, BGR, or BGRA images")
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            image.save(Path(tmp_dir) / f"{index:08d}.png")

        final_path = Path(tmp_dir) / "final.mp4"
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", str(Path(tmp_dir) / "%08d.png"),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(final_path),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(final_path, output_path)
            return output_path
        video_bytes = final_path.read_bytes()

    b64 = base64.b64encode(video_bytes).decode("utf-8")

    # Unique id so multiple calls to show_video in one notebook don't collide
    uid = uuid.uuid4().hex[:8]
    video_id = f"vid_{uid}"
    caption_id = f"cap_{uid}"

    if labels is not None:
        labels_json = json.dumps([str(l) for l in labels])
        caption_html = f'<div id="{caption_id}" style="width:{width}px; text-align:center; font-family:sans-serif; font-size:14px; color:#fff; margin-top:6px; min-height:1.2em;"></div>'
        script = f'''
        <script>
        (function() {{
            const labels = {labels_json};
            const fps = {fps};
            const video = document.getElementById("{video_id}");
            const caption = document.getElementById("{caption_id}");

            function updateCaption() {{
                if (!video.duration) return;
                let idx = Math.floor(video.currentTime * fps);
                if (idx < 0) idx = 0;
                if (idx >= labels.length) idx = labels.length - 1;
                caption.textContent = labels[idx];
            }}

            video.addEventListener("timeupdate", updateCaption);
            video.addEventListener("seeked", updateCaption);
            video.addEventListener("loadedmetadata", updateCaption);
        }})();
        </script>
        '''
    else:
        caption_html = ""
        script = ""

    video_tag = f'''
    <div style="width:{width}px;">
      <video id="{video_id}" width="{width}" height="{height}" controls autoplay loop>
        <source src="data:video/mp4;base64,{b64}" type="video/mp4">
      </video>
      {caption_html}
    </div>
    {script}
    '''
    display(HTML(video_tag))


def main():
    parser = argparse.ArgumentParser(description="Make an MP4 from PNG files sorted by filename.")
    parser.add_argument("image_dir", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "081_SPKMVLFLP")
    parser.add_argument("--fps", type=float, default=24)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    if not args.image_dir.is_dir():
        parser.error(f"image directory does not exist: {args.image_dir}")
    paths = sorted(p for p in args.image_dir.iterdir()
                   if p.is_file() and p.suffix.lower() == ".png")
    if not paths:
        parser.error(f"no PNG images found in {args.image_dir}")
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB"))[:, :, ::-1])
    output = args.output or args.image_dir.with_suffix(".mp4")
    try:
        make_movie(frames, fps=args.fps, output_path=output)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Saved {len(frames)} frames at {args.fps:g} fps to {output}")


if __name__ == "__main__":
    main()
