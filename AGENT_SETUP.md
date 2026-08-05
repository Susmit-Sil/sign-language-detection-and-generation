# Agent Environment Setup Guide

> [!IMPORTANT]
> **Instructions for the AI Assistant / Agent:**
> You are helping the user set up this project on a new computer. Follow the steps below carefully to prevent dependency conflicts, especially with library-heavy environments like PyTorch, TensorFlow, and Ultralytics.

---

## Workspace Layout
The repository contains two distinct Python environments:
1. **Root Environment (`/venv`)**: Runs streamlit, Facenet-PyTorch, etc. Configured by [requirements.txt](file:///e:/Hackathon/requirements.txt).
2. **Object Detection Environment (`/RealTimeObjectDetection/signlang_venv`)**: Runs YOLO/Ultralytics, OpenCV WebRTC, and TensorFlow. Configured by [RealTimeObjectDetection/requirements.txt](file:///e:/Hackathon/RealTimeObjectDetection/requirements.txt).

---

## 🛠️ Execution Steps for the Agent

Please execute the following sequence of commands to set up the environments perfectly.

### Step 1: Recreate the Root Virtual Environment
Run these commands to initialize and install the root project dependencies:
```powershell
# Create venv
python -m venv venv

# Activate and install dependencies
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Recreate the Object Detection Virtual Environment
Run these commands to build the specialized environment for YOLO and TensorFlow:
```powershell
# Create sub venv
python -m venv RealTimeObjectDetection/signlang_venv

# Activate and install dependencies
.\RealTimeObjectDetection\signlang_venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r RealTimeObjectDetection/requirements.txt
```

### Step 3: Fetching the Heavy Datasets
If the `asl_datasets/` directory or other heavy files are missing:
1. Ask the user for their Hugging Face username/repo name where the datasets were uploaded.
2. Run the following command in the main virtual environment to pull the dataset files directly:
   ```powershell
   huggingface-cli download --repo-type dataset <username>/<dataset-repo> --local-dir asl_datasets
   ```
