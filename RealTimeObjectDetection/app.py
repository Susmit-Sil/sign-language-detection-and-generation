"""
🤟 Sign Language Detection — Real-Time Streamlit App
Detects 5 sign language gestures using YOLOv8:
  Hello | I Love You | No | Thank You | Yes
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
import os
from collections import deque

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🤟 Sign Language Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15));
        border-radius: 20px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 400;
    }

    /* Detection Card */
    .detection-card {
        background: linear-gradient(145deg, rgba(30, 27, 75, 0.8), rgba(30, 41, 59, 0.8));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .detection-card:hover {
        border-color: rgba(168, 85, 247, 0.5);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
    }

    .gesture-emoji {
        font-size: 4rem;
        margin-bottom: 0.5rem;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    .gesture-name {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .confidence-value {
        font-size: 1.1rem;
        color: #a5b4fc;
        font-weight: 500;
        margin-top: 0.25rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #c4b5fd;
    }

    /* Sign Reference Cards */
    .sign-ref {
        background: rgba(30, 27, 75, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        transition: all 0.2s ease;
    }
    .sign-ref:hover {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .sign-ref .emoji { font-size: 1.8rem; }
    .sign-ref .label {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* History */
    .history-item {
        background: rgba(30,27,75,0.5);
        border-left: 3px solid #818cf8;
        border-radius: 0 8px 8px 0;
        padding: 0.5rem 0.75rem;
        margin: 0.35rem 0;
        color: #cbd5e1;
        font-size: 0.85rem;
        font-family: 'Inter', monospace;
    }

    /* Stats */
    .stat-box {
        background: linear-gradient(145deg, rgba(30, 27, 75, 0.7), rgba(49, 46, 129, 0.4));
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-box .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #a5b4fc;
    }
    .stat-box .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .status-live {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: #4ade80;
    }
    .status-idle {
        background: rgba(234, 179, 8, 0.15);
        border: 1px solid rgba(234, 179, 8, 0.4);
        color: #facc15;
    }

    /* Video container */
    .video-container {
        border-radius: 16px;
        overflow: hidden;
        border: 2px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }

    /* Remove default Streamlit padding */
    .block-container { padding-top: 1rem; }

    /* Footer */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.75rem;
        padding: 1rem 0;
        border-top: 1px solid rgba(99, 102, 241, 0.1);
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Gesture Metadata ───────────────────────────────────────────────────────────
GESTURE_INFO = {
    "Hello":      {"emoji": "👋", "desc": "Open hand wave"},
    "I Love You": {"emoji": "🤟", "desc": "Extended thumb, index & pinky"},
    "No":         {"emoji": "✋", "desc": "Closed fist shake"},
    "Thank You":  {"emoji": "🙏", "desc": "Flat hand from chin outward"},
    "Yes":        {"emoji": "👍", "desc": "Fist pump nod"},
}

COLORS = {
    "Hello":      (72, 199, 142),   # teal
    "I Love You": (167, 139, 250),  # purple
    "No":         (248, 113, 113),  # red
    "Thank You":  (96, 165, 250),   # blue
    "Yes":        (251, 191, 36),   # amber
}


# ─── Model Loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained YOLOv8 model. Prioritizes augmented model."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Prioritize augmented model (better webcam generalization)
    possible_paths = [
        os.path.join(base_dir, "runs", "detect", "sign_language_aug", "weights", "best.pt"),
        os.path.join(base_dir, "runs", "detect", "sign_language3", "weights", "best.pt"),
        os.path.join(base_dir, "runs", "detect", "sign_language2", "weights", "best.pt"),
        os.path.join(base_dir, "runs", "detect", "sign_language", "weights", "best.pt"),
        os.path.join(base_dir, "best.pt"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            model = YOLO(path)
            return model, path

    return None, None


def draw_detections(frame, results, min_conf=0.4):
    """Draw bounding boxes and labels on frame with custom styling."""
    detections = []
    if results and len(results) > 0:
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < min_conf:
                    continue
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = COLORS.get(class_name, (255, 255, 255))

                # Draw filled rectangle background for label
                label = f"{class_name} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                # Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Label background
                cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 10, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Corner accents
                corner_len = 20
                cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
                cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 4)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 4)
                cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 4)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 4)
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4)

                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                })

    return frame, detections


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤟 Sign Reference")
    for gesture, info in GESTURE_INFO.items():
        st.markdown(f"""
        <div class="sign-ref">
            <span class="emoji">{info['emoji']}</span>
            <div>
                <span class="label">{gesture}</span><br>
                <span style="color:#64748b;font-size:0.8rem">{info['desc']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## ⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold", 0.05, 0.95, 0.25, 0.05,
        help="Lower = more detections (may include false positives). Default 25% works best for webcam."
    )
    max_detections = st.slider(
        "Max Detections", 1, 10, 5,
        help="Maximum simultaneous detections"
    )

    st.markdown("---")
    st.markdown("## 📊 Detection History")
    history_container = st.container()


# ─── Main Content ───────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🤟 Sign Language Detector</h1>
    <p>Real-time ASL gesture recognition powered by YOLOv8</p>
</div>
""", unsafe_allow_html=True)

# Load model
model, model_path = load_model()

if model is None:
    st.error("""
    ### ⚠️ No trained model found!
    
    Please train the model first by running:
    ```
    signlang_venv\\Scripts\\python.exe train.py
    ```
    
    The training script will create `runs/detect/sign_language/weights/best.pt`.
    Once training is complete, restart this app.
    """)
    st.stop()

# Model info bar
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">YOLOv8n</div>
        <div class="stat-label">Model Architecture</div>
    </div>
    """, unsafe_allow_html=True)
with col_info2:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">5</div>
        <div class="stat-label">Gesture Classes</div>
    </div>
    """, unsafe_allow_html=True)
with col_info3:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{confidence_threshold:.0%}</div>
        <div class="stat-label">Confidence Threshold</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Mode Selection ──────────────────────────────────────────────────────────────
tab_webcam, tab_upload = st.tabs(["🎥 Live Webcam", "📁 Test with Image"])

# ─── Image Upload Tab ────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("Upload an image to test if the model can detect signs correctly.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        results = model(img, verbose=False, conf=confidence_threshold, max_det=max_detections)
        annotated, dets = draw_detections(img.copy(), results, confidence_threshold)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, channels="RGB", use_container_width=True)
        if dets:
            for d in dets:
                info = GESTURE_INFO.get(d['class'], {'emoji': '❓'})
                st.success(f"{info['emoji']} **{d['class']}** — Confidence: {d['confidence']:.1%}")
        else:
            st.warning("No signs detected. Try lowering the confidence threshold in the sidebar.")

# ─── Webcam Tab ──────────────────────────────────────────────────────────────────
with tab_webcam:
    col_video, col_result = st.columns([3, 1])

    with col_result:
        st.markdown("### 🎯 Current Detection")
        detection_display = st.empty()
        detection_display.markdown("""
        <div class="detection-card">
            <div class="gesture-emoji">🔍</div>
            <div class="gesture-name">Waiting...</div>
            <div class="confidence-value">Show a sign to start</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fps_display = st.empty()

    with col_video:
        run_detection = st.toggle("🎥 Start Camera", value=False, help="Toggle webcam on/off")

        if run_detection:
            st.markdown('<span class="status-badge status-live">● LIVE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-idle">● IDLE</span>', unsafe_allow_html=True)

        video_placeholder = st.empty()

# ─── Detection Loop ─────────────────────────────────────────────────────────────
if run_detection:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        st.error("❌ Could not open webcam. Please check your camera connection.")
        st.stop()

    history = deque(maxlen=15)
    frame_count = 0
    fps_start = time.time()

    try:
        while run_detection:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)  # Mirror
            frame_count += 1

            # Run detection
            results = model(frame, verbose=False, conf=confidence_threshold, max_det=max_detections)
            annotated_frame, detections = draw_detections(frame, results, confidence_threshold)

            # Convert BGR → RGB for Streamlit
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display video
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update detection display
            if detections:
                best = max(detections, key=lambda d: d["confidence"])
                info = GESTURE_INFO.get(best["class"], {"emoji": "❓", "desc": ""})
                detection_display.markdown(f"""
                <div class="detection-card">
                    <div class="gesture-emoji">{info['emoji']}</div>
                    <div class="gesture-name">{best['class']}</div>
                    <div class="confidence-value">Confidence: {best['confidence']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

                # Add to history
                timestamp = time.strftime("%H:%M:%S")
                history.appendleft(f"{timestamp} — {info['emoji']} {best['class']} ({best['confidence']:.0%})")
            else:
                detection_display.markdown("""
                <div class="detection-card">
                    <div class="gesture-emoji">🔍</div>
                    <div class="gesture-name">No Sign Detected</div>
                    <div class="confidence-value">Show a gesture to the camera</div>
                </div>
                """, unsafe_allow_html=True)

            # FPS calculation
            elapsed = time.time() - fps_start
            if elapsed > 0:
                fps = frame_count / elapsed
                fps_display.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">{fps:.1f}</div>
                    <div class="stat-label">FPS</div>
                </div>
                """, unsafe_allow_html=True)

            # Update history in sidebar
            if history:
                with history_container:
                    history_html = "".join(
                        f'<div class="history-item">{item}</div>' for item in list(history)[:10]
                    )
                    st.markdown(history_html, unsafe_allow_html=True)

    finally:
        cap.release()

else:
    # Show placeholder when camera is off
    video_placeholder.markdown("""
    <div style="
        background: linear-gradient(145deg, rgba(30,27,75,0.6), rgba(30,41,59,0.6));
        border: 2px dashed rgba(99,102,241,0.3);
        border-radius: 16px;
        padding: 4rem 2rem;
        text-align: center;
    ">
        <p style="font-size: 3rem; margin-bottom: 0.5rem;">📸</p>
        <p style="color: #94a3b8; font-size: 1.1rem; font-weight: 500;">
            Toggle <b>"Start Camera"</b> above to begin detection
        </p>
        <p style="color: #64748b; font-size: 0.85rem;">
            Make sure your webcam is connected and accessible
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit & YOLOv8 · Sign Language Detection System
</div>
""", unsafe_allow_html=True)
