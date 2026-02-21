"""
wlasl_loader.py
───────────────
Download WLASL sign language videos and extract frames.
Supports full video clip extraction (all frames) for motion GIFs,
plus single-frame extraction for thumbnails.
"""

from __future__ import annotations

import json
import os
import subprocess
import numpy as np
import requests
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
WLASL_JSON = BASE_DIR / "asl_datasets" / "wlasl" / "start_kit" / "WLASL_v0.3.json"
WLASL_COMPLETE_JSON = BASE_DIR / "asl_datasets" / "wlasl-complete" / "WLASL_v0.3.json"
WLASL_COMPLETE_VIDEOS = BASE_DIR / "asl_datasets" / "wlasl-complete" / "videos"
VIDEO_CACHE_DIR = BASE_DIR / "asl_datasets" / "wlasl_videos"
FRAME_CACHE_DIR = BASE_DIR / "asl_datasets" / "wlasl_frames"

_wlasl_data = None
_word_to_urls = None
_word_to_video_ids = None  # Maps gloss → list of video_id strings


def _load_wlasl_json():
    global _wlasl_data, _word_to_urls, _word_to_video_ids
    if _wlasl_data is not None:
        return

    # Try the complete dataset first, then the starter kit
    json_path = WLASL_COMPLETE_JSON if WLASL_COMPLETE_JSON.exists() else WLASL_JSON

    if not json_path.exists():
        _wlasl_data = []
        _word_to_urls = {}
        _word_to_video_ids = {}
        return

    with open(json_path, "r") as f:
        _wlasl_data = json.load(f)

    _word_to_urls = {}
    _word_to_video_ids = {}

    for entry in _wlasl_data:
        gloss = entry["gloss"].upper()
        urls = []
        video_ids = []

        for inst in entry.get("instances", []):
            video_id = inst.get("video_id", "")
            if video_id:
                video_ids.append({
                    "video_id": video_id,
                    "frame_start": inst.get("frame_start", -1),
                    "frame_end": inst.get("frame_end", -1),
                })

            url = inst.get("url", "")
            if not url:
                continue
            if url.endswith(".swf") or "aslbricks" in url:
                continue
            urls.append({
                "url": url,
                "video_id": video_id,
                "frame_start": inst.get("frame_start", -1),
                "frame_end": inst.get("frame_end", -1),
                "is_youtube": "youtube.com" in url or "youtu.be" in url,
            })

        urls.sort(key=lambda x: (x["is_youtube"], x["url"]))
        if urls:
            _word_to_urls[gloss] = urls
        if video_ids:
            _word_to_video_ids[gloss] = video_ids

    print(f"WLASL loaded: {len(_word_to_urls)} words, {len(_word_to_video_ids)} with local videos")


def has_word_sign(word: str) -> bool:
    _load_wlasl_json()
    word_upper = word.upper()
    # Check local videos first, then remote URLs
    return word_upper in _word_to_video_ids or word_upper in _word_to_urls


def _download_direct(url: str, save_path: Path) -> bool:
    """Download a direct URL (non-YouTube)."""
    try:
        resp = requests.get(url, timeout=15, stream=True)
        resp.raise_for_status()
        os.makedirs(save_path.parent, exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception:
        return False


def _download_youtube(url: str, save_path: Path) -> bool:
    """Download a YouTube video via yt-dlp."""
    try:
        cmd = [
            "yt-dlp",
            "-f", "best[ext=mp4]",
            "-o", str(save_path),
            "--no-warnings",
            "--quiet",
            "--no-playlist",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and save_path.exists()
    except Exception:
        return False


def _find_local_video(word: str) -> Path | None:
    """Find a local video file for the given word from the WLASL-complete dataset."""
    _load_wlasl_json()
    word_upper = word.upper()

    # Check pre-downloaded named videos first (e.g., HELLO.mp4)
    named_video = VIDEO_CACHE_DIR / f"{word_upper}.mp4"
    if named_video.exists():
        return named_video

    # Check WLASL-complete videos by video_id
    if word_upper in _word_to_video_ids:
        for vid_info in _word_to_video_ids[word_upper]:
            vid_id = vid_info["video_id"]
            video_path = WLASL_COMPLETE_VIDEOS / f"{vid_id}.mp4"
            if video_path.exists():
                return video_path

    return None


def _get_frame_range(word: str) -> tuple[int, int]:
    """Get the frame_start, frame_end for a word from WLASL metadata."""
    _load_wlasl_json()
    word_upper = word.upper()

    if word_upper in _word_to_video_ids:
        for vid_info in _word_to_video_ids[word_upper]:
            fs = vid_info.get("frame_start", -1)
            fe = vid_info.get("frame_end", -1)
            if fs >= 0 and fe >= 0:
                return fs, fe

    if word_upper in _word_to_urls:
        for url_info in _word_to_urls[word_upper]:
            fs = url_info.get("frame_start", -1)
            fe = url_info.get("frame_end", -1)
            if fs >= 0 and fe >= 0:
                return fs, fe

    return -1, -1


def _enhance_image(img: Image.Image) -> Image.Image:
    """Enhance image for better clarity."""
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)
    return img


# ═════════════════════════════════════════════════════════════════════════════
# Multi-frame extraction (for motion GIFs)
# ═════════════════════════════════════════════════════════════════════════════

def _extract_all_frames(video_path: Path, frame_start: int = -1,
                        frame_end: int = -1, target_fps: int = 12,
                        max_frames: int = 45) -> list[np.ndarray]:
    """
    Extract frames from a video file within [frame_start, frame_end].
    Subsamples to approximately target_fps.  Returns a list of BGR ndarrays.

    Args:
        video_path: Path to the MP4 file.
        frame_start, frame_end: WLASL-annotated frame range (-1 = use all).
        target_fps: Desired output frame rate (used for subsampling).
        max_frames: Hard cap on how many frames to return per word.

    Returns:
        List of BGR numpy arrays (one per extracted frame).
    """
    if not HAS_CV2:
        return []

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        if total <= 0:
            cap.release()
            return []

        # Determine range
        start = max(0, frame_start) if frame_start >= 0 else 0
        end = min(frame_end, total - 1) if frame_end >= 0 else total - 1

        # Subsample: pick every Nth frame to approximate target_fps
        step = max(1, int(round(src_fps / target_fps)))
        frame_indices = list(range(start, end + 1, step))

        # Cap at max_frames
        if len(frame_indices) > max_frames:
            # Evenly subsample
            indices = np.linspace(0, len(frame_indices) - 1, max_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in indices]

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()
        return frames

    except Exception:
        return []


def get_word_sign_frames(word: str, size: int = 512) -> list[Image.Image]:
    """
    Get ALL frames of a person signing the given ASL word — for motion GIFs.

    Tries local videos first (WLASL-complete + cached), then downloads.

    Returns:
        List of PIL Images (multiple frames showing the sign in motion).
        Returns empty list if no video is found.
    """
    _load_wlasl_json()
    word_upper = word.upper()

    # ── Try local video ───────────────────────────────────────────────────
    video_path = _find_local_video(word_upper)
    frame_start, frame_end = _get_frame_range(word_upper)

    if video_path and video_path.exists():
        raw_frames = _extract_all_frames(video_path, frame_start, frame_end)
        if raw_frames:
            pil_frames = []
            for bgr in raw_frames:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img = img.resize((size, size), Image.Resampling.LANCZOS)
                img = _enhance_image(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                pil_frames.append(img)
            return pil_frames

    # ── Try downloading ───────────────────────────────────────────────────
    if word_upper in _word_to_urls:
        for url_info in _word_to_urls[word_upper][:3]:
            url = url_info["url"]
            is_yt = url_info["is_youtube"]
            video_dl_path = VIDEO_CACHE_DIR / f"{word_upper}.mp4"

            if not video_dl_path.exists():
                os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)
                print(f"  Downloading {word_upper}...")
                success = _download_youtube(url, video_dl_path) if is_yt else _download_direct(url, video_dl_path)
                if not success:
                    if video_dl_path.exists():
                        video_dl_path.unlink()
                    continue

            fs = url_info.get("frame_start", -1)
            fe = url_info.get("frame_end", -1)
            raw_frames = _extract_all_frames(video_dl_path, fs, fe)
            if raw_frames:
                pil_frames = []
                for bgr in raw_frames:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    img = img.resize((size, size), Image.Resampling.LANCZOS)
                    img = _enhance_image(img)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    pil_frames.append(img)
                return pil_frames
            else:
                if video_dl_path.exists():
                    video_dl_path.unlink()

    return []


# ═════════════════════════════════════════════════════════════════════════════
# Single-frame extraction (backward compat)
# ═════════════════════════════════════════════════════════════════════════════

def _calculate_sharpness(frame):
    if not HAS_CV2:
        return 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def get_word_sign_image(word: str, size: int = 512) -> Image.Image:
    """Get a single image of a person signing the given ASL word (backward compat)."""
    # Try to get frames and pick the sharpest one
    frames = get_word_sign_frames(word, size)
    if not frames:
        return None

    if len(frames) == 1:
        return frames[0]

    # Pick the sharpest frame from the middle third
    mid_start = len(frames) // 3
    mid_end = 2 * len(frames) // 3
    mid_frames = frames[mid_start:mid_end] if mid_end > mid_start else frames

    best = None
    best_score = -1
    for img in mid_frames:
        if HAS_CV2:
            arr = np.array(img)
            score = _calculate_sharpness(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        else:
            score = 0
        if score > best_score:
            best_score = score
            best = img

    return best or frames[len(frames) // 2]


if __name__ == "__main__":
    test_words = ["HELLO", "I", "LOVE", "YOU", "BOOK"]
    print("Testing WLASL loader (multi-frame)...")
    for w in test_words:
        frames = get_word_sign_frames(w)
        if frames:
            print(f"  {w}: {len(frames)} frames, first={frames[0].size}")
        else:
            single = get_word_sign_image(w)
            status = f"single frame {single.size}" if single else "failed"
            print(f"  {w}: no video frames, {status}")
