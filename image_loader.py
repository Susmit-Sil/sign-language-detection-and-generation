"""
image_loader.py
───────────────
Unified ASL image/frame loader — word-level video clips first, fingerspelling fallback.

Priority:
1. WLASL word-level video clip (e.g., "HELLO" → multiple frames of person signing)
2. ASL Alphabet fingerspelling (e.g., unknown word → repeated frames per letter)

Special handling:
- Letters J and Z require motion in real ASL.
  We load real J.gif / Z.gif sign-language GIFs from the dataset folder
  and return all their frames as the motion sequence.
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from wlasl_loader import has_word_sign, get_word_sign_image, get_word_sign_frames

# ─── Dataset paths ────────────────────────────────────────────────────────────
DATASETS_DIR = Path(__file__).parent / "asl_datasets"
ASL_ALPHABET_DIR = DATASETS_DIR / "asl_alphabet_train" / "asl_alphabet_train"
AYURAJ_ASL_DIR = DATASETS_DIR / "ayuraj_asl"

# How many times to repeat each static fingerspelled letter frame.
# At 83ms/frame (12fps), 18 repeats ≈ 1.5 seconds per letter — readable.
LETTER_HOLD_FRAMES = 18

# Target frame count for J / Z motion GIFs after subsampling.
# 24 frames × 83ms ≈ 2 seconds — fast enough to feel natural, slow enough to follow.
MOTION_LETTER_TARGET_FRAMES = 24

# Number of animation frames for J and Z motion
MOTION_LETTER_FRAMES = 10


# ═════════════════════════════════════════════════════════════════════════════
# Letter-level loaders (fingerspelling fallback)
# ═════════════════════════════════════════════════════════════════════════════

def _load_letter_asl_alphabet(letter: str) -> Image.Image:
    """Load a random letter image from ASL Alphabet dataset."""
    letter_dir = ASL_ALPHABET_DIR / letter.upper()
    if not letter_dir.exists():
        return None
    images = list(letter_dir.glob("*.jpg")) + list(letter_dir.glob("*.png"))
    if not images:
        return None
    img = Image.open(random.choice(images))
    return img.resize((512, 512), Image.Resampling.LANCZOS)


def _load_letter_ayuraj(letter: str) -> Image.Image:
    """Load a letter image from Ayuraj ASL dataset."""
    for subdir in ["asl_dataset", ""]:
        letter_dir = AYURAJ_ASL_DIR / subdir / letter.upper() if subdir else AYURAJ_ASL_DIR / letter.upper()
        if letter_dir.exists():
            images = list(letter_dir.glob("*.jpg")) + list(letter_dir.glob("*.png")) + list(letter_dir.glob("*.jpeg"))
            if images:
                img = Image.open(random.choice(images))
                return img.resize((512, 512), Image.Resampling.LANCZOS)
    return None


def _load_letter_image(letter: str) -> Image.Image:
    """Load an image for a single letter, trying all datasets."""
    img = _load_letter_asl_alphabet(letter)
    if img:
        return img
    img = _load_letter_ayuraj(letter)
    if img:
        return img
    return _create_placeholder(letter)


def _create_placeholder(text: str) -> Image.Image:
    """Create a simple placeholder with text (last resort)."""
    img = Image.new('RGB', (512, 512), color='white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 120)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (512 - (bbox[2] - bbox[0])) // 2
    y = (512 - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, fill='black', font=font)
    return img


# ═════════════════════════════════════════════════════════════════════════════
# Motion-letter loaders for J and Z (real sign language GIFs)
# ═════════════════════════════════════════════════════════════════════════════

def _load_gif_frames(gif_path: Path, size: int = 512,
                     target_frames: int | None = None) -> list[Image.Image]:
    """
    Extract every frame from a GIF file and return them as a list of PIL Images
    resized to *size* × *size* RGB.

    If *target_frames* is given, subsample the extracted frames to that count
    so the animation plays at a consistent speed regardless of the GIF's
    original frame count.

    Falls back to an empty list if the file doesn't exist or can't be opened.
    """
    if not gif_path.exists():
        return []
    try:
        gif = Image.open(gif_path)
        raw = []
        while True:
            frame = gif.copy().convert('RGB')
            frame = frame.resize((size, size), Image.Resampling.LANCZOS)
            raw.append(frame)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass  # Reached the last frame — normal exit
    except Exception:
        return []

    if not raw:
        return []

    # Subsample to target_frames if needed
    if target_frames and len(raw) > target_frames:
        import numpy as np
        indices = np.linspace(0, len(raw) - 1, target_frames, dtype=int)
        raw = [raw[i] for i in indices]

    return raw


def _create_j_motion_frames(size: int = 512) -> list[Image.Image]:
    """Load real ASL J sign frames from J.gif, subsampled to ~2 seconds."""
    gif_path = ASL_ALPHABET_DIR / "J.gif"
    frames = _load_gif_frames(gif_path, size, target_frames=MOTION_LETTER_TARGET_FRAMES)
    if frames:
        return frames
    img = _load_letter_asl_alphabet('J') or _create_placeholder('J')
    return [img] * LETTER_HOLD_FRAMES


def _create_z_motion_frames(size: int = 512) -> list[Image.Image]:
    """Load real ASL Z sign frames from Z.gif, subsampled to ~2 seconds."""
    gif_path = ASL_ALPHABET_DIR / "Z.gif"
    frames = _load_gif_frames(gif_path, size, target_frames=MOTION_LETTER_TARGET_FRAMES)
    if frames:
        return frames
    img = _load_letter_asl_alphabet('Z') or _create_placeholder('Z')
    return [img] * LETTER_HOLD_FRAMES


# ═════════════════════════════════════════════════════════════════════════════
# Main API — single image (backward compat)
# ═════════════════════════════════════════════════════════════════════════════

def get_sign_image(word: str) -> list[Image.Image]:
    """
    Get ASL sign image(s) for a word.  Returns a list with usually 1 image
    for word signs or N images for fingerspelled letters.
    """
    word_upper = word.upper().strip()

    if has_word_sign(word_upper):
        img = get_word_sign_image(word_upper)
        if img:
            return [img]

    # Fallback: fingerspell
    letters = [c for c in word_upper if c.isalpha()]
    images = []
    for letter in letters:
        img = _load_letter_image(letter)
        if img:
            images.append(img)

    return images if images else [_create_placeholder(word_upper)]


# ═════════════════════════════════════════════════════════════════════════════
# New API — full motion frames
# ═════════════════════════════════════════════════════════════════════════════

def get_sign_frames(word: str) -> list[Image.Image]:
    """
    Get ALL frames for an ASL sign — returns a full motion clip.

    Strategy:
    1. If WLASL has a video → return all extracted frames (15-45 frames)
    2. Otherwise, fingerspell → each letter image repeated LETTER_HOLD_FRAMES times
       - Special: J and Z get animated motion frames instead of static repeats

    Returns:
        List of PIL Images forming a smooth animation.
    """
    word_upper = word.upper().strip()

    # ── Try word-level video first (full motion) ──────────────────────────
    if has_word_sign(word_upper):
        frames = get_word_sign_frames(word_upper)
        if frames and len(frames) > 1:
            return frames

    # ── Fallback: fingerspell with held frames ────────────────────────────
    letters = [c for c in word_upper if c.isalpha()]
    all_frames = []

    for letter in letters:
        # ── J and Z: use motion frames (they require movement in ASL) ────
        if letter == 'J':
            all_frames.extend(_create_j_motion_frames())
            continue
        if letter == 'Z':
            all_frames.extend(_create_z_motion_frames())
            continue

        # ── All other letters: static image, held for several frames ─────
        img = _load_letter_image(letter)
        if img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            for _ in range(LETTER_HOLD_FRAMES):
                all_frames.append(img)

    if not all_frames:
        placeholder = _create_placeholder(word_upper)
        all_frames = [placeholder] * LETTER_HOLD_FRAMES

    return all_frames


if __name__ == "__main__":
    test_words = ["HELLO", "I", "LOVE", "YOU", "GO", "JAZZ", "ENJOY"]
    print("Testing unified frame loader...")
    for word in test_words:
        frames = get_sign_frames(word)
        method = "video clip" if has_word_sign(word.upper()) else "fingerspelling"
        print(f"  {word}: {len(frames)} frames via {method}")
