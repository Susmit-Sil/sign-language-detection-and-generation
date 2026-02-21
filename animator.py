"""
animator.py
───────────
Combine all .png frames into a single smooth animated GIF.

Reads the metadata.json produced by generator.py to know which frames
belong to which word, and inserts brief title-card frames at the
start of each word's clip.
"""

import os
import glob
import json
from PIL import Image, ImageDraw, ImageFont


# ── Config ────────────────────────────────────────────────────────────────────
FRAME_DURATION_MS = 83       # ~12 fps for motion frames
TITLE_DURATION_MS = 600      # How long the word title card shows
TITLE_HOLD_FRAMES = 4        # Number of title card frames (TITLE_DURATION_MS / FRAME_DURATION_MS)
TRANSITION_FRAMES = 2        # Brief pause between words
CANVAS_SIZE = 512


def _create_title_card(word: str, size: int = CANVAS_SIZE) -> Image.Image:
    """Create a dark title card showing the current word."""
    img = Image.new('RGB', (size, size), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    # Main word
    try:
        font_big = ImageFont.truetype("arial.ttf", 72)
        font_small = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = font_big

    text = word.upper()
    bbox = draw.textbbox((0, 0), text, font=font_big)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (size - tw) // 2
    y = (size - th) // 2 - 20

    # Draw text with accent color
    draw.text((x, y), text, fill=(163, 130, 255), font=font_big)

    # Subtitle
    sub = "ASL Sign"
    bbox2 = draw.textbbox((0, 0), sub, font=font_small)
    sw = bbox2[2] - bbox2[0]
    sx = (size - sw) // 2
    sy = y + th + 20
    draw.text((sx, sy), sub, fill=(148, 163, 184), font=font_small)

    return img


def _create_transition_frame(size: int = CANVAS_SIZE) -> Image.Image:
    """Create a brief dark transition frame between words."""
    return Image.new('RGB', (size, size), color=(15, 15, 20))


def create_asl_gif(image_folder: str = "temp_frames",
                   output_path: str = "asl_output.gif") -> str:
    """
    Combine all .png frames into a single animated GIF with title cards.

    Reads metadata.json from generator.py to group frames by word
    and insert title cards between each word's clip.

    Returns:
        The output file path of the created GIF.
    """
    # Load metadata
    metadata_file = os.path.join(image_folder, "metadata.json")
    word_ranges = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            meta = json.load(f)
            word_ranges = meta.get("word_ranges", {})

    # Collect and sort PNGs by frame index
    pattern = os.path.join(image_folder, "*.png")
    files = sorted(
        glob.glob(pattern),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split("_")[1]),
    )

    if not files:
        raise FileNotFoundError(f"No .png files found in '{image_folder}'")

    # Build ordered word list from metadata
    ordered_words = sorted(word_ranges.items(), key=lambda x: x[1]["start"])

    output_frames = []
    durations = []

    if ordered_words:
        for word, rng in ordered_words:
            start = rng["start"]
            end = rng["end"]

            # ── Title card ────────────────────────────────────────────────
            title = _create_title_card(word)
            for _ in range(TITLE_HOLD_FRAMES):
                output_frames.append(title)
                durations.append(FRAME_DURATION_MS)

            # ── Motion frames ─────────────────────────────────────────────
            for idx in range(start, end + 1):
                fname = f"frame_{idx:05d}.png"
                fpath = os.path.join(image_folder, fname)
                if os.path.exists(fpath):
                    img = Image.open(fpath).convert("RGB")
                    output_frames.append(img)
                    durations.append(FRAME_DURATION_MS)

            # ── Transition pause ──────────────────────────────────────────
            trans = _create_transition_frame()
            for _ in range(TRANSITION_FRAMES):
                output_frames.append(trans)
                durations.append(FRAME_DURATION_MS)
    else:
        # No metadata — fallback: just stitch frames sequentially
        for f in files:
            img = Image.open(f).convert("RGB")
            output_frames.append(img)
            durations.append(FRAME_DURATION_MS)

    if not output_frames:
        raise ValueError("No frames to animate")

    # Save as animated GIF
    output_frames[0].save(
        output_path,
        save_all=True,
        append_images=output_frames[1:],
        duration=durations,
        loop=0,
    )

    total_secs = sum(durations) / 1000
    print(f"GIF saved -> {output_path}  ({len(output_frames)} frames, {total_secs:.1f}s)")
    return output_path


if __name__ == "__main__":
    create_asl_gif()
