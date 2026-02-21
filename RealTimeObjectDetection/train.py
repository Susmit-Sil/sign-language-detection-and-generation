"""
Train YOLOv8 on the sign language dataset with HEAVY augmentation.
This version uses aggressive augmentation to help generalize from 
a small dataset to live webcam input.
"""

from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML = os.path.join(BASE_DIR, "datasets", "sign_language", "dataset.yaml")

def main():
    # Load YOLOv8 nano pretrained on COCO
    model = YOLO("yolov8n.pt")

    # Train with aggressive augmentation for small dataset generalization
    results = model.train(
        data=DATASET_YAML,
        epochs=200,           # More epochs for small dataset
        imgsz=640,
        batch=8,
        name="sign_language_aug",
        patience=50,          # Longer patience
        save=True,
        plots=True,
        device="0",           # RTX 4060 GPU

        # --- Heavy Augmentation ---
        augment=True,
        hsv_h=0.02,           # Hue shift
        hsv_s=0.7,            # Saturation shift
        hsv_v=0.5,            # Value/brightness shift
        degrees=15.0,         # Rotation ±15°
        translate=0.2,        # Translation ±20%
        scale=0.5,            # Scale ±50%
        shear=5.0,            # Shear ±5°
        perspective=0.001,    # Perspective warp
        flipud=0.1,           # Vertical flip 10%
        fliplr=0.5,           # Horizontal flip 50%
        mosaic=1.0,           # Mosaic augmentation 100%
        mixup=0.3,            # MixUp augmentation 30%
        copy_paste=0.2,       # Copy-paste augmentation 20%
        erasing=0.3,          # Random erasing 30%

        workers=2,
        project=os.path.join(BASE_DIR, "runs", "detect"),
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("Best model: runs/detect/sign_language_aug/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
