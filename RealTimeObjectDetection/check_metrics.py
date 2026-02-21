"""Check training metrics and test inference on known images."""
import csv
import glob
import os

# 1. Print training metrics
print("=" * 60)
print("TRAINING METRICS")
print("=" * 60)

csv_path = r"E:\Hackathon\RealTimeObjectDetection\runs\detect\sign_language3\results.csv"
with open(csv_path) as f:
    reader = csv.reader(f)
    rows = list(reader)
    header = [c.strip() for c in rows[0]]

    # Find metric columns
    map50_idx = None
    map50_95_idx = None
    prec_idx = None
    recall_idx = None
    for i, h in enumerate(header):
        if "mAP50(B)" in h and "95" not in h:
            map50_idx = i
        elif "mAP50-95" in h:
            map50_95_idx = i
        elif "precision" in h:
            prec_idx = i
        elif "recall" in h:
            recall_idx = i

    print(f"Found columns: mAP50={map50_idx}, mAP50-95={map50_95_idx}, P={prec_idx}, R={recall_idx}")
    print()

    for row in rows[-10:]:
        vals = [v.strip() for v in row]
        epoch = vals[0]
        m50 = vals[map50_idx] if map50_idx else "?"
        m95 = vals[map50_95_idx] if map50_95_idx else "?"
        p = vals[prec_idx] if prec_idx else "?"
        r = vals[recall_idx] if recall_idx else "?"
        print(f"  Epoch {epoch:>3s}: mAP50={m50}  mAP50-95={m95}  P={p}  R={r}")


# 2. Test inference on known images
print()
print("=" * 60)
print("INFERENCE TEST (conf=0.05)")
print("=" * 60)

from ultralytics import YOLO

model_path = r"E:\Hackathon\RealTimeObjectDetection\runs\detect\sign_language3\weights\best.pt"
model = YOLO(model_path)

test_images = glob.glob(r"E:\Hackathon\RealTimeObjectDetection\Tensorflow\workspace\images\train\*.jpg")[:10]
for img_path in test_images:
    results = model(img_path, verbose=False, conf=0.05)
    dets = results[0].boxes
    basename = os.path.basename(img_path)
    if len(dets) == 0:
        print(f"  {basename}: NO DETECTIONS")
    else:
        for box in dets:
            cls = results[0].names[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f"  {basename}: {cls} ({conf:.2%})")
