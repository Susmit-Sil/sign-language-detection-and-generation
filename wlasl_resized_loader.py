"""
wlasl_resized_loader.py
───────────────────────
Load pre-extracted WLASL-2000 frames from the resized dataset.
Much faster than downloading videos - instant image access!
"""

import os
import glob
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

BASE_DIR = Path(__file__).parent
WLASL_RESIZED_DIR = BASE_DIR / "asl_datasets" / "wlasl-complete"

_word_to_images = None


def _load_word_index():
    """Build index of word -> image paths."""
    global _word_to_images
    if _word_to_images is not None:
        return
    
    _word_to_images = {}
    
    if not WLASL_RESIZED_DIR.exists():
        print("WLASL-2000 Resized dataset not found")
        return
    
    # Each word has a subdirectory with image files
    for word_dir in WLASL_RESIZED_DIR.iterdir():
        if not word_dir.is_dir():
            continue
        
        word = word_dir.name.upper()
        
        # Get all images in this word's directory
        image_files = list(word_dir.glob("*.jpg")) + list(word_dir.glob("*.png"))
        
        if image_files:
            _word_to_images[word] = image_files
    
    print(f"WLASL-2000 Resized loaded: {len(_word_to_images)} words")


def has_word_sign(word: str) -> bool:
    """Check if we have pre-extracted frames for this word."""
    _load_word_index()
    return word.upper() in _word_to_images


def get_word_sign_image(word: str, size: int = 512) -> Image.Image:
    """Get a pre-extracted image for the given ASL word."""
    _load_word_index()
    word_upper = word.upper()
    
    if word_upper not in _word_to_images:
        return None
    
    image_paths = _word_to_images[word_upper]
    
    if not image_paths:
        return None
    
    # Pick a random image (or first one for consistency)
    img_path = random.choice(image_paths)
    
    try:
        img = Image.open(img_path)
        
        # Enhance the image for better visibility
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
        
        # Increase contrast slightly
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        
        # Resize to target size
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        
        # Ensure RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None


if __name__ == "__main__":
    print("Testing WLASL-2000 Resized loader...")
    test_words = ["HELLO", "I", "LOVE", "YOU", "BOOK", "WALK"]
    for word in test_words:
        has_it = has_word_sign(word)
        if has_it:
            img = get_word_sign_image(word)
            status = f"✅ {img.size if img else 'failed'}"
        else:
            status = "❌ not in dataset"
        print(f"  {word}: {status}")
