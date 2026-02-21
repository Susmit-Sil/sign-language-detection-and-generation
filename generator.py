"""
generator.py
────────────
Generate ASL sign frame sequences from ASL glosses.

For each word:
- If it has a WLASL video → extract full motion clip (multiple frames)
- If it's rare → fingerspell with held letter images
- All frames saved to temp_frames/ with metadata.

The metadata tracks frame ranges per word so the animator
can add title cards and stitch them together.
"""

import os
import shutil
import json
from image_loader import get_sign_frames

_frame_counter = 0
_word_ranges = {}  # Maps word → {"start": int, "end": int}
FRAMES_DIR = "temp_frames"
METADATA_FILE = "temp_frames/metadata.json"


def generate_sign_frames(word: str) -> list[str]:
    """
    Generate ALL frames for one ASL gloss word.

    Returns the list of saved frame paths.
    """
    global _frame_counter, _word_ranges

    frames = get_sign_frames(word)

    if not frames:
        raise ValueError(f"No frames found for word: {word}")

    os.makedirs(FRAMES_DIR, exist_ok=True)

    start_idx = _frame_counter
    saved_paths = []

    for img in frames:
        filepath = os.path.join(FRAMES_DIR, f"frame_{_frame_counter:05d}.png")
        # Ensure RGB before saving
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(filepath)
        saved_paths.append(filepath)
        _frame_counter += 1

    end_idx = _frame_counter - 1

    # Save word range metadata
    _word_ranges[word.upper()] = {"start": start_idx, "end": end_idx}

    # Write metadata file
    metadata = {
        "word_ranges": _word_ranges,
        "total_frames": _frame_counter,
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    return saved_paths


# Keep backward-compatible alias
def generate_sign_image(word: str) -> str:
    """Backward-compatible: generates frames and returns path to first frame."""
    paths = generate_sign_frames(word)
    return paths[0]


def clear_frames():
    """Remove all frames and reset counters."""
    global _frame_counter, _word_ranges
    _frame_counter = 0
    _word_ranges = {}
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR, exist_ok=True)


if __name__ == "__main__":
    clear_frames()
    print("Testing generator (multi-frame)...")
    for gloss in ["HELLO", "I", "LOVE", "YOU"]:
        try:
            paths = generate_sign_frames(gloss)
            print(f"  [{gloss}] → {len(paths)} frames")
        except Exception as e:
            print(f"  [{gloss}] Error: {e}")
