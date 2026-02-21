"""
Convert Pascal VOC XML annotations to YOLO format.
Creates the dataset directory structure needed for YOLOv8 training.
"""

import os
import xml.etree.ElementTree as ET
import shutil
import glob

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "Tensorflow", "workspace", "images", "train")
TEST_IMG_DIR = os.path.join(BASE_DIR, "Tensorflow", "workspace", "images", "test")

# Output dataset directory
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "sign_language")

# Class mapping (must match label_map.pbtxt order, 0-indexed for YOLO)
CLASS_MAP = {
    "Hello": 0,
    "I Love You": 1,
    "No": 2,
    "Thank You": 3,
    "Yes": 4,
}


def convert_voc_to_yolo(xml_path, class_map):
    """Parse a Pascal VOC XML file and return YOLO-format lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in class_map:
            print(f"  WARNING: Unknown class '{class_name}' in {xml_path}, skipping")
            continue

        class_id = class_map[class_name]
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Convert to YOLO format: x_center, y_center, width, height (all normalized)
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return yolo_lines


def process_split(src_dir, split_name):
    """Process all XML files in a directory and create YOLO dataset split."""
    img_out = os.path.join(DATASET_DIR, "images", split_name)
    lbl_out = os.path.join(DATASET_DIR, "labels", split_name)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    xml_files = glob.glob(os.path.join(src_dir, "*.xml"))
    print(f"\n[{split_name.upper()}] Found {len(xml_files)} XML files in {src_dir}")

    converted = 0
    for xml_path in xml_files:
        yolo_lines = convert_voc_to_yolo(xml_path, CLASS_MAP)
        if not yolo_lines:
            continue

        # Get corresponding image file
        base_name = os.path.splitext(os.path.basename(xml_path))[0]
        img_src = os.path.join(src_dir, base_name + ".jpg")
        if not os.path.exists(img_src):
            img_src = os.path.join(src_dir, base_name + ".png")
        if not os.path.exists(img_src):
            print(f"  WARNING: Image not found for {xml_path}, skipping")
            continue

        # Copy image
        shutil.copy2(img_src, os.path.join(img_out, os.path.basename(img_src)))

        # Write YOLO label file
        txt_name = base_name + ".txt"
        with open(os.path.join(lbl_out, txt_name), "w") as f:
            f.write("\n".join(yolo_lines))

        converted += 1

    print(f"  Converted {converted} annotations to YOLO format")
    return converted


def create_dataset_yaml():
    """Create the dataset.yaml config file for YOLOv8."""
    yaml_content = f"""# Sign Language Detection Dataset
path: {DATASET_DIR.replace(os.sep, '/')}
train: images/train
val: images/val

# Classes
names:
  0: Hello
  1: I Love You
  2: No
  3: Thank You
  4: Yes

nc: 5
"""
    yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nCreated dataset config: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    print("=" * 60)
    print("Pascal VOC → YOLO Converter for Sign Language Dataset")
    print("=" * 60)

    # Process train and test (test → val for YOLO convention)
    process_split(TRAIN_IMG_DIR, "train")
    process_split(TEST_IMG_DIR, "val")

    # Create YAML config
    yaml_path = create_dataset_yaml()

    print("\n" + "=" * 60)
    print("DONE! Dataset ready at:", DATASET_DIR)
    print(f"YAML config: {yaml_path}")
    print("=" * 60)
