"""
🤟 GenAI ASL Communicator — Unified Streamlit App
  Mode 1: Text → ASL  (type a sentence, get an animated GIF)
  Mode 2: ASL → Text  (real-time webcam sign language detection via YOLOv8)
"""

import os
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI ASL Communicator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global */
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    }
    .block-container { padding-top: 1rem; }

    /* ── Shared header ── */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15));
        border-radius: 20px;
        border: 1px solid rgba(99,102,241,0.2);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    .main-header h1 {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .main-header p { color: #94a3b8; font-size: 1rem; }

    /* ── Text-to-ASL styles ── */
    .gif-container { display: flex; justify-content: center; margin-top: 1.5rem; }
    .stImage > img {
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .rejected-card {
        background: linear-gradient(135deg, #1e1b2e, #2d1f3d);
        border: 1px solid #6b21a8; border-radius: 16px;
        padding: 2.5rem 2rem; text-align: center;
        margin: 2rem auto; max-width: 500px;
        box-shadow: 0 4px 20px rgba(107,33,168,0.3);
    }
    .rejected-icon  { font-size: 4rem; margin-bottom: 1rem; }
    .rejected-title  { color: #e879f9; font-size: 1.4rem; font-weight: 700; margin-bottom: 0.8rem; }
    .rejected-reason { color: #c4b5fd; font-size: 1rem; line-height: 1.6; }
    .rejected-hint   { color: #94a3b8; font-size: 0.85rem; margin-top: 1.2rem; }

    /* ── Recognition styles ── */
    .detection-card {
        background: linear-gradient(145deg, rgba(30,27,75,0.8), rgba(30,41,59,0.8));
        border: 1px solid rgba(99,102,241,0.3); border-radius: 16px;
        padding: 1.5rem; text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .detection-card:hover {
        border-color: rgba(168,85,247,0.5);
        box-shadow: 0 12px 40px rgba(99,102,241,0.15);
    }
    .gesture-emoji {
        font-size: 4rem; margin-bottom: 0.5rem;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.1)} }
    .gesture-name {
        font-size: 1.8rem; font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .confidence-value { font-size: 1.1rem; color: #a5b4fc; font-weight: 500; margin-top: 0.25rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(99,102,241,0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 { color: #c4b5fd; }

    .sign-ref {
        background: rgba(30,27,75,0.6); border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px; padding: 0.75rem 1rem; margin: 0.5rem 0;
        display: flex; align-items: center; gap: 0.75rem;
        transition: all 0.2s ease;
    }
    .sign-ref:hover { background: rgba(99,102,241,0.15); border-color: rgba(99,102,241,0.4); }
    .sign-ref .emoji { font-size: 1.8rem; }
    .sign-ref .label { color: #e2e8f0; font-weight: 600; font-size: 0.95rem; }

    .history-item {
        background: rgba(30,27,75,0.5); border-left: 3px solid #818cf8;
        border-radius: 0 8px 8px 0; padding: 0.5rem 0.75rem;
        margin: 0.35rem 0; color: #cbd5e1; font-size: 0.85rem;
    }

    .stat-box {
        background: linear-gradient(145deg, rgba(30,27,75,0.7), rgba(49,46,129,0.4));
        border: 1px solid rgba(99,102,241,0.25); border-radius: 12px;
        padding: 1rem; text-align: center;
    }
    .stat-box .stat-value { font-size: 1.6rem; font-weight: 700; color: #a5b4fc; }
    .stat-box .stat-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }

    .status-badge {
        display: inline-block; padding: 0.3rem 1rem; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600; letter-spacing: 0.03em;
    }
    .status-live { background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.4); color: #4ade80; }
    .status-idle { background: rgba(234,179,8,0.15); border: 1px solid rgba(234,179,8,0.4); color: #facc15; }

    .video-container {
        border-radius: 16px; overflow: hidden;
        border: 2px solid rgba(99,102,241,0.3);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    .footer {
        text-align: center; color: #475569; font-size: 0.75rem;
        padding: 1rem 0; border-top: 1px solid rgba(99,102,241,0.1); margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Mode selector + contextual controls
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤟 GenAI ASL Communicator")
    app_mode = st.radio(
        "Choose Mode",
        ["✍️ Text → ASL (Generate)", "📷 ASL → Text (Recognize)"],
        index=0,
        help="Switch between generating ASL animations and recognizing signs from webcam.",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1 — TEXT → ASL  (GIF Generator)
# ═══════════════════════════════════════════════════════════════════════════════
if app_mode.startswith("✍️"):

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🤟 Text → ASL Generator</h1>
        <p>Type an English sentence and watch it come alive in American Sign Language</p>
    </div>
    """, unsafe_allow_html=True)

    # Lazy imports (only needed for this mode)
    from translator import translate_to_gloss, ContentRejectedError
    from generator import generate_sign_frames, clear_frames, FRAMES_DIR
    from animator import create_asl_gif

    # ── Input ────────────────────────────────────────────────────────────────
    user_input = st.text_input("Enter your message here", placeholder="e.g. I love you")
    generate_btn = st.button("🚀 Generate", use_container_width=True)

    OUTPUT_GIF = "asl_output.gif"

    # ── Pipeline ─────────────────────────────────────────────────────────────
    if generate_btn and user_input.strip():
        clear_frames()
        if os.path.exists(OUTPUT_GIF):
            os.remove(OUTPUT_GIF)

        try:
            # Step 1 – Translate
            with st.status("Translating to ASL gloss...", expanded=True) as status:
                glosses = translate_to_gloss(user_input)
                st.write(f"**ASL Glosses:** {glosses}")
                status.update(label="Translation complete ✅", state="complete")

            # Step 2 – Generate frames
            with st.status("Extracting sign video clips...", expanded=True) as status:
                progress = st.progress(0)
                total_frames = 0
                for idx, word in enumerate(glosses):
                    st.write(f"Loading motion clip for **{word}**...")
                    paths = generate_sign_frames(word)
                    total_frames += len(paths)
                    progress.progress((idx + 1) / len(glosses))
                st.write(f"**Total frames:** {total_frames}")
                status.update(label=f"Extracted {total_frames} frames ✅", state="complete")

            # Step 3 – Animate
            with st.status("Creating animated GIF...", expanded=True) as status:
                create_asl_gif(image_folder=FRAMES_DIR, output_path=OUTPUT_GIF)
                status.update(label="GIF ready ✅", state="complete")

            # Step 4 – Display
            st.markdown("---")
            st.markdown('<div class="gif-container">', unsafe_allow_html=True)
            st.image(OUTPUT_GIF, caption="ASL Translation", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        except ContentRejectedError as e:
            st.markdown("---")
            st.markdown(f"""
            <div class="rejected-card">
                <div class="rejected-icon">🚫</div>
                <div class="rejected-title">Content Not Available</div>
                <div class="rejected-reason">{e.reason}</div>
                <div class="rejected-hint">Please try a different sentence. This tool is designed for everyday communication.</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    elif generate_btn:
        st.warning("Please enter a sentence first.")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2 — ASL → TEXT  (YOLOv8 Real-Time Recognition)
# ═══════════════════════════════════════════════════════════════════════════════
else:
    import cv2
    import numpy as np
    import time
    from collections import deque
    from ultralytics import YOLO

    # ── Gesture metadata ─────────────────────────────────────────────────────
    GESTURE_INFO = {
        "Hello":      {"emoji": "👋", "desc": "Open hand wave"},
        "I Love You": {"emoji": "🤟", "desc": "Extended thumb, index & pinky"},
        "No":         {"emoji": "✋", "desc": "Closed fist shake"},
        "Thank You":  {"emoji": "🙏", "desc": "Flat hand from chin outward"},
        "Yes":        {"emoji": "👍", "desc": "Fist pump nod"},
    }

    COLORS = {
        "Hello":      (72, 199, 142),
        "I Love You": (167, 139, 250),
        "No":         (248, 113, 113),
        "Thank You":  (96, 165, 250),
        "Yes":        (251, 191, 36),
    }

    # ── Model loading ────────────────────────────────────────────────────────
    @st.cache_resource
    def load_yolo_model():
        """Load the trained YOLOv8 model from the RealTimeObjectDetection subfolder."""
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RealTimeObjectDetection")

        possible_paths = [
            os.path.join(base_dir, "runs", "detect", "sign_language_aug", "weights", "best.pt"),
            os.path.join(base_dir, "runs", "detect", "sign_language3", "weights", "best.pt"),
            os.path.join(base_dir, "runs", "detect", "sign_language2", "weights", "best.pt"),
            os.path.join(base_dir, "runs", "detect", "sign_language", "weights", "best.pt"),
            os.path.join(base_dir, "best.pt"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return YOLO(path), path

        return None, None

    def draw_detections(frame, results, min_conf=0.4, scale=1.0):
        """Draw bounding boxes and labels on frame with custom styling.
        
        Args:
            frame:   Full-resolution display frame to draw on.
            results: YOLO results (may have been run on a smaller frame).
            min_conf: Minimum confidence threshold.
            scale:   Multiply bbox coords by this factor to map back to display frame size.
        """
        detections = []
        if results and len(results) > 0:
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < min_conf:
                        continue
                    cls_id = int(box.cls[0])
                    class_name = result.names[cls_id]
                    # Scale coords back to original frame dimensions
                    x1, y1, x2, y2 = (int(v * scale) for v in box.xyxy[0])
                    color = COLORS.get(class_name, (255, 255, 255))

                    label = f"{class_name} {conf:.0%}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    # Corner accents
                    cl = 20
                    cv2.line(frame, (x1, y1), (x1 + cl, y1), color, 4)
                    cv2.line(frame, (x1, y1), (x1, y1 + cl), color, 4)
                    cv2.line(frame, (x2, y1), (x2 - cl, y1), color, 4)
                    cv2.line(frame, (x2, y1), (x2, y1 + cl), color, 4)
                    cv2.line(frame, (x1, y2), (x1 + cl, y2), color, 4)
                    cv2.line(frame, (x1, y2), (x1, y2 - cl), color, 4)
                    cv2.line(frame, (x2, y2), (x2 - cl, y2), color, 4)
                    cv2.line(frame, (x2, y2), (x2, y2 - cl), color, 4)

                    detections.append({"class": class_name, "confidence": conf, "bbox": (x1, y1, x2, y2)})
        return frame, detections

    # ── Sidebar extras for recognition mode ──────────────────────────────────
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
        max_detections = st.slider("Max Detections", 1, 10, 5, help="Maximum simultaneous detections")

        st.markdown("---")
        st.markdown("## 📊 Detection History")
        history_container = st.container()

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>📷 ASL → Text Recognizer</h1>
        <p>Real-time ASL gesture recognition powered by YOLOv8</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ───────────────────────────────────────────────────────────
    model, model_path = load_yolo_model()

    if model is None:
        st.error("""
        ### ⚠️ No trained model found!

        Expected model weights in `RealTimeObjectDetection/runs/detect/` folder.
        Please make sure the trained model exists, then restart this app.
        """)
        st.stop()

    # ── Model info bar ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value">YOLOv8n</div>
            <div class="stat-label">Model Architecture</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value">5</div>
            <div class="stat-label">Gesture Classes</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{confidence_threshold:.0%}</div>
            <div class="stat-label">Confidence Threshold</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs: webcam vs image upload ─────────────────────────────────────────
    tab_webcam, tab_upload = st.tabs(["🎥 Live Webcam", "📁 Test with Image"])

    # ── Image upload ─────────────────────────────────────────────────────────
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

    # ── Webcam live detection ────────────────────────────────────────────────
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

    # ── Detection loop ───────────────────────────────────────────────────────
    TARGET_FPS   = 20                          # cap to avoid memory buildup
    FRAME_DELAY  = 1.0 / TARGET_FPS
    INFER_WIDTH  = 416                         # resize before inference (faster)
    HISTORY_EVERY = 30                         # update sidebar only every N frames

    if run_detection:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # keep the capture buffer small so we always get the latest frame
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            st.error("❌ Could not open webcam. Please check your camera connection.")
            st.stop()

        history = deque(maxlen=15)
        frame_count = 0
        fps_window_start = time.time()
        last_history_update = -1

        try:
            while run_detection:
                loop_start = time.time()

                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Failed to read frame from webcam.")
                    break

                frame = cv2.flip(frame, 1)
                frame_count += 1

                # ── Resize for faster inference, keep original for display ──
                h, w = frame.shape[:2]
                scale = INFER_WIDTH / w
                small = cv2.resize(frame, (INFER_WIDTH, int(h * scale)))

                results = model(small, verbose=False, conf=confidence_threshold, max_det=max_detections)

                # Scale bounding boxes back to original frame size
                annotated_frame, detections = draw_detections(frame.copy(), results, confidence_threshold, scale=1.0/scale)

                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # ── Detection card ───────────────────────────────────────────
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

                # ── FPS counter (rolling window) ─────────────────────────────
                now = time.time()
                elapsed = now - fps_window_start
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    fps_display.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-value">{fps:.1f}</div>
                        <div class="stat-label">FPS</div>
                    </div>
                    """, unsafe_allow_html=True)
                    frame_count = 0
                    fps_window_start = now

                # ── Sidebar history — update every HISTORY_EVERY frames only ─
                if history and (frame_count % HISTORY_EVERY == 0 or last_history_update == -1):
                    with history_container:
                        history_html = "".join(
                            f'<div class="history-item">{item}</div>' for item in list(history)[:10]
                        )
                        st.markdown(history_html, unsafe_allow_html=True)
                    last_history_update = frame_count

                # ── FPS throttle: sleep remaining time to hit TARGET_FPS ─────
                spent = time.time() - loop_start
                sleep_time = FRAME_DELAY - spent
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            cap.release()

    else:
        video_placeholder.markdown("""
        <div style="
            background: linear-gradient(145deg, rgba(30,27,75,0.6), rgba(30,41,59,0.6));
            border: 2px dashed rgba(99,102,241,0.3);
            border-radius: 16px; padding: 4rem 2rem; text-align: center;
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


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit · GenAI ASL Communicator — Text↔Sign Language
</div>
""", unsafe_allow_html=True)
