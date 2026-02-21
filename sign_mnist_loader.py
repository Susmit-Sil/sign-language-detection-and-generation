"""
sign_mnist_loader.py
────────────────────
Load Sign Language MNIST dataset for letter-level ASL signs.
28x28 grayscale images of ASL alphabet (A-Z, excluding J and Z which require motion).
"""

import os
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path

BASE_DIR = Path(__file__).parent
MNIST_TRAIN_CSV = BASE_DIR / "asl_datasets" / "sign_mnist_train.csv"
MNIST_TEST_CSV = BASE_DIR / "asl_datasets" / "sign_mnist_test.csv"

# MNIST labels: 0-25 = A-Z (excluding J=9, Z=25)
# J and Z are excluded because they require motion
_LABEL_TO_LETTER = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

_LETTER_TO_LABEL = {v: k for k, v in _LABEL_TO_LETTER.items()}

_train_df = None
_test_df = None


def _load_data():
    """Load MNIST CSV data into memory (cached)."""
    global _train_df, _test_df
    if _train_df is not None:
        return
    
    if MNIST_TRAIN_CSV.exists():
        _train_df = pd.read_csv(MNIST_TRAIN_CSV)
        print(f"Sign MNIST train loaded: {len(_train_df)} samples")
    else:
        _train_df = pd.DataFrame()
    
    if MNIST_TEST_CSV.exists():
        _test_df = pd.read_csv(MNIST_TEST_CSV)
        print(f"Sign MNIST test loaded: {len(_test_df)} samples")
    else:
        _test_df = pd.DataFrame()


def has_letter_sign(letter: str) -> bool:
    """Check if we have a sign for this letter."""
    _load_data()
    letter_upper = letter.upper()
    if letter_upper in ['J', 'Z']:  # Motion letters not in MNIST
        return False
    return letter_upper in _LETTER_TO_LABEL


def get_letter_sign_image(letter: str, size: int = 512) -> Image.Image:
    """Get an MNIST image for the given ASL letter."""
    _load_data()
    letter_upper = letter.upper()
    
    if not has_letter_sign(letter_upper):
        return None
    
    label = _LETTER_TO_LABEL[letter_upper]
    
    # Get all samples for this letter from train set
    samples = _train_df[_train_df['label'] == label]
    
    if len(samples) == 0:
        return None
    
    # Pick a random sample (or first one for consistency)
    sample = samples.iloc[np.random.randint(0, len(samples))]
    
    # Extract pixel values (784 pixels = 28x28)
    pixels = sample.drop('label').values.astype(np.uint8)
    
    # Reshape to 28x28
    img_array = pixels.reshape(28, 28)
    
    # Convert to PIL Image
    img = Image.fromarray(img_array, mode='L')
    
    # Enhance the image for better visibility
    # Invert colors (MNIST has white hand on black background)
    img = Image.eval(img, lambda x: 255 - x)
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    # Resize to target size with good interpolation
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    
    # Convert to RGB
    img = img.convert('RGB')
    
    return img


if __name__ == "__main__":
    print("Testing Sign MNIST loader...")
    test_letters = ['A', 'B', 'C', 'H', 'I', 'J', 'Z']
    for letter in test_letters:
        has_it = has_letter_sign(letter)
        if has_it:
            img = get_letter_sign_image(letter)
            status = f"✅ {img.size if img else 'failed'}"
        else:
            status = "❌ not available (motion letter)"
        print(f"  {letter}: {status}")
