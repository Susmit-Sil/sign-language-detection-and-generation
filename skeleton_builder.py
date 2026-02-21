"""
skeleton_builder.py
───────────────────
Generates OpenPose-style hand skeleton images for ASL signs.

Each skeleton is a 512×512 black image with coloured lines connecting
21 hand landmarks (wrist + 4 joints per finger).

Landmark index map (MediaPipe / OpenPose convention):
  0  = Wrist
  1- 4 = Thumb   (CMC → MCP → IP → TIP)
  5- 8 = Index   (MCP → PIP → DIP → TIP)
  9-12 = Middle  (MCP → PIP → DIP → TIP)
 13-16 = Ring    (MCP → PIP → DIP → TIP)
 17-20 = Pinky   (MCP → PIP → DIP → TIP)
"""

import cv2
import numpy as np
from typing import Optional

# ── OpenPose hand connections ────────────────────────────────────────────────
HAND_CONNECTIONS = [
    # Wrist → each finger base
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index
    (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
]

# Colours per finger group (BGR for OpenCV)
FINGER_COLORS = {
    "wrist":  (128, 128, 128),   # grey
    "thumb":  (0, 0, 255),       # red
    "index":  (0, 165, 255),     # orange
    "middle": (0, 255, 255),     # yellow
    "ring":   (0, 255, 0),       # green
    "pinky":  (255, 0, 0),       # blue
}

def _color_for_connection(a: int, b: int) -> tuple:
    """Return the colour for the bone connecting landmarks *a* and *b*."""
    pair = {a, b}
    if pair & {1, 2, 3, 4}:
        return FINGER_COLORS["thumb"]
    if pair & {5, 6, 7, 8}:
        return FINGER_COLORS["index"]
    if pair & {9, 10, 11, 12}:
        return FINGER_COLORS["middle"]
    if pair & {13, 14, 15, 16}:
        return FINGER_COLORS["ring"]
    if pair & {17, 18, 19, 20}:
        return FINGER_COLORS["pinky"]
    return FINGER_COLORS["wrist"]


# ═════════════════════════════════════════════════════════════════════════════
# ASL POSE LIBRARY
# ─────────────────────────────────────────────────────────────────────────────
# Each entry maps an ASL gloss word to a list of 21 (x, y) landmark tuples
# on a 512×512 canvas.  Coordinates are hand-tuned to reflect the correct
# hand shape for each sign.
#
# ── Anatomy reference (right hand, palm facing viewer) ──
#   • Thumb  = landmarks  1–4
#   • Index  = landmarks  5–8
#   • Middle = landmarks  9–12
#   • Ring   = landmarks 13–16
#   • Pinky  = landmarks 17–20
# ═════════════════════════════════════════════════════════════════════════════
ASL_POSE_LIBRARY: dict[str, list[tuple[int, int]]] = {

    # ── "I LOVE YOU" sign ────────────────────────────────────────────────
    # Thumb OUT, Index UP, Middle FOLDED, Ring FOLDED, Pinky UP
    "I LOVE YOU": [
        (256, 400),   # 0  Wrist
        # Thumb – extended out to the right
        (210, 370),   # 1
        (170, 340),   # 2
        (130, 300),   # 3
        (100, 260),   # 4  Thumb tip
        # Index – straight up
        (230, 330),   # 5
        (220, 260),   # 6
        (215, 190),   # 7
        (210, 120),   # 8  Index tip
        # Middle – folded into palm
        (256, 330),   # 9
        (256, 290),   # 10
        (256, 310),   # 11
        (256, 340),   # 12 Middle tip (curled)
        # Ring – folded into palm
        (280, 330),   # 13
        (285, 290),   # 14
        (285, 310),   # 15
        (280, 340),   # 16 Ring tip (curled)
        # Pinky – straight up
        (305, 340),   # 17
        (315, 270),   # 18
        (320, 200),   # 19
        (325, 135),   # 20 Pinky tip
    ],

    # ── Letter I ─────────────────────────────────────────────────────────
    # Only pinky UP, all others folded into a fist
    "I": [
        (256, 400),   # 0  Wrist
        # Thumb – folded across palm
        (220, 370),   # 1
        (200, 340),   # 2
        (220, 310),   # 3
        (245, 310),   # 4  Thumb tip (resting over fist)
        # Index – folded
        (230, 330),   # 5
        (225, 300),   # 6
        (235, 320),   # 7
        (245, 340),   # 8
        # Middle – folded
        (256, 330),   # 9
        (256, 295),   # 10
        (258, 315),   # 11
        (256, 340),   # 12
        # Ring – folded
        (280, 330),   # 13
        (282, 295),   # 14
        (280, 315),   # 15
        (278, 340),   # 16
        # Pinky – straight up
        (305, 340),   # 17
        (315, 270),   # 18
        (320, 200),   # 19
        (325, 135),   # 20 Pinky tip
    ],

    # ── LOVE ─────────────────────────────────────────────────────────────
    # Crossed arms over chest gesture – we show an open hand (all fingers up)
    # as the "hand shape" component before crossing
    "LOVE": [
        (256, 420),   # 0  Wrist
        # Thumb – spread out
        (190, 370),   # 1
        (155, 330),   # 2
        (125, 290),   # 3
        (100, 255),   # 4
        # Index – up
        (225, 330),   # 5
        (215, 260),   # 6
        (208, 195),   # 7
        (200, 130),   # 8
        # Middle – up
        (256, 320),   # 9
        (256, 245),   # 10
        (256, 175),   # 11
        (256, 110),   # 12
        # Ring – up
        (287, 325),   # 13
        (295, 250),   # 14
        (300, 185),   # 15
        (305, 120),   # 16
        # Pinky – up
        (315, 340),   # 17
        (328, 270),   # 18
        (335, 210),   # 19
        (340, 145),   # 20
    ],

    # ── YOU (pointing) ───────────────────────────────────────────────────
    # Index pointing forward/up, all others folded
    "YOU": [
        (256, 400),   # 0  Wrist
        # Thumb – wrapped over fist
        (215, 365),   # 1
        (195, 335),   # 2
        (215, 310),   # 3
        (240, 305),   # 4
        # Index – extended straight up
        (230, 330),   # 5
        (220, 260),   # 6
        (215, 190),   # 7
        (210, 120),   # 8  Index tip
        # Middle – folded
        (256, 330),   # 9
        (256, 295),   # 10
        (258, 315),   # 11
        (256, 340),   # 12
        # Ring – folded
        (280, 330),   # 13
        (282, 295),   # 14
        (280, 315),   # 15
        (278, 340),   # 16
        # Pinky – folded
        (305, 340),   # 17
        (310, 315),   # 18
        (305, 330),   # 19
        (300, 345),   # 20
    ],

    # ── HELLO (open palm wave) ───────────────────────────────────────────
    "HELLO": [
        (256, 430),   # 0  Wrist
        # Thumb – spread out
        (185, 380),   # 1
        (145, 340),   # 2
        (115, 295),   # 3
        (90, 260),    # 4
        # Index – up
        (222, 335),   # 5
        (210, 265),   # 6
        (202, 198),   # 7
        (195, 130),   # 8
        # Middle – up
        (256, 325),   # 9
        (256, 250),   # 10
        (256, 180),   # 11
        (256, 110),   # 12
        # Ring – up
        (290, 330),   # 13
        (298, 255),   # 14
        (304, 190),   # 15
        (308, 125),   # 16
        # Pinky – up
        (320, 345),   # 17
        (332, 275),   # 18
        (340, 215),   # 19
        (345, 150),   # 20
    ],

    # ── YES (fist nodding) ───────────────────────────────────────────────
    "YES": [
        (256, 400),   # 0  Wrist
        # Thumb – wrapped over
        (215, 365),   # 1
        (195, 335),   # 2
        (205, 308),   # 3
        (230, 300),   # 4
        # Index – folded
        (230, 330),   # 5
        (225, 300),   # 6
        (235, 315),   # 7
        (245, 335),   # 8
        # Middle – folded
        (256, 325),   # 9
        (256, 295),   # 10
        (258, 312),   # 11
        (256, 335),   # 12
        # Ring – folded
        (280, 325),   # 13
        (282, 295),   # 14
        (280, 312),   # 15
        (278, 335),   # 16
        # Pinky – folded
        (305, 335),   # 17
        (308, 310),   # 18
        (305, 325),   # 19
        (300, 340),   # 20
    ],

    # ── NO (index + middle snap together) ────────────────────────────────
    "NO": [
        (256, 400),   # 0  Wrist
        # Thumb – out, meets index+middle
        (210, 365),   # 1
        (180, 330),   # 2
        (170, 290),   # 3
        (175, 255),   # 4
        # Index – extended up
        (230, 330),   # 5
        (220, 265),   # 6
        (215, 200),   # 7
        (210, 140),   # 8
        # Middle – extended up alongside index
        (256, 325),   # 9
        (256, 258),   # 10
        (256, 195),   # 11
        (256, 135),   # 12
        # Ring – folded
        (280, 330),   # 13
        (282, 300),   # 14
        (280, 318),   # 15
        (278, 338),   # 16
        # Pinky – folded
        (305, 340),   # 17
        (308, 315),   # 18
        (305, 328),   # 19
        (300, 345),   # 20
    ],

    # ── THANK YOU (open hand from chin) ──────────────────────────────────
    "THANK-YOU": [
        (256, 430),   # 0  Wrist
        (185, 380),   # 1
        (145, 340),   # 2
        (115, 295),   # 3
        (90, 260),    # 4
        (222, 335),   # 5
        (210, 265),   # 6
        (202, 198),   # 7
        (195, 130),   # 8
        (256, 325),   # 9
        (256, 250),   # 10
        (256, 180),   # 11
        (256, 110),   # 12
        (290, 330),   # 13
        (298, 255),   # 14
        (304, 190),   # 15
        (308, 125),   # 16
        (320, 345),   # 17
        (332, 275),   # 18
        (340, 215),   # 19
        (345, 150),   # 20
    ],
}


# ── Fallback: generic open-hand pose for unknown words ───────────────────
_DEFAULT_POSE: list[tuple[int, int]] = ASL_POSE_LIBRARY["HELLO"]


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════
def create_asl_skeleton(word: str, size: int = 512) -> np.ndarray:
    """
    Return a *size*×*size* BGR image with a hand skeleton drawn in OpenPose
    style for the given ASL gloss *word*.

    If the word is not in the pose library the default open-hand is used.

    Args:
        word: ASL gloss word (case-insensitive).
        size: Canvas dimension in pixels (default 512).

    Returns:
        A numpy array (H, W, 3) dtype uint8 – the skeleton image.
    """
    landmarks = ASL_POSE_LIBRARY.get(word.upper(), _DEFAULT_POSE)

    # Scale landmarks if the canvas differs from the default 512
    if size != 512:
        scale = size / 512
        landmarks = [(int(x * scale), int(y * scale)) for x, y in landmarks]

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    # Draw bones
    for a, b in HAND_CONNECTIONS:
        colour = _color_for_connection(a, b)
        pt1 = landmarks[a]
        pt2 = landmarks[b]
        cv2.line(canvas, pt1, pt2, colour, thickness=4, lineType=cv2.LINE_AA)

    # Draw joints
    for idx, (x, y) in enumerate(landmarks):
        colour = _color_for_connection(idx, idx)
        cv2.circle(canvas, (x, y), 6, colour, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 6, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return canvas


if __name__ == "__main__":
    # Quick test – save skeleton images for the library words
    import os
    os.makedirs("skeleton_previews", exist_ok=True)
    for word in ASL_POSE_LIBRARY:
        img = create_asl_skeleton(word)
        safe = word.replace(" ", "_")
        path = f"skeleton_previews/{safe}.png"
        cv2.imwrite(path, img)
        print(f"  [{word}] → {path}")
    print("Done.")
