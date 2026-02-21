# Sign Language Detection and Generation

This project features a real-time American Sign Language (ASL) detection and generation system built with Python, YOLOv8, and Streamlit. It unifies two core capabilities: capturing webcam video to detect sign language hand gestures in real-time, and taking text input to generate corresponding animated ASL signs.

## Features
- **Real-Time Detection:** Uses YOLOv8 to detect ASL alphabets from live webcam feed.
- **Text-to-ASL Generation:** Converts input text into a sequence of ASL gestures displayed as an animated GIF.
- **Interactive UI:** A unified Streamlit interface allows users to easily toggle between detection and generation modes.
- **Custom Model Training:** Includes scripts for preparing datasets, augmenting data, and training YOLOv8 models from scratch.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Susmit-Sil/sign-language-detection-and-generation.git
   cd sign-language-detection-and-generation
   ```

2. **Run the Initialization Script:**
   The easiest way to get started is by running the `run_app.bat` script, which automatically sets up the Python virtual environment, installs dependencies, and launches the app.
   ```cmd
   run_app.bat
   ```

3. **Alternatively, manually set up the environment:**
   ```bash
   python -m venv signlang_venv
   signlang_venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Project Structure
- `app.py`: The main Streamlit application script.
- `run_app.bat`: Windows batch script for automated environment setup and application launch.
- `requirements.txt`: List of required Python packages.
- `image_loader.py` / `wlasl_loader.py` / `sign_mnist_loader.py`: Utilities for handling various sign language datasets.
- `translator.py` / `generator.py` / `animator.py` / `skeleton_builder.py`: Core modules for the text-to-ASL generation pipeline.
- `download_datasets.py`: Script to download required training and testing datasets.

## Requirements
- Python 3.9+
- A working webcam (for real-time detection)
- CUDA-compatible GPU (optional, but highly recommended for fast YOLOv8 inference and training)

## How to Compile/Run the App
Simply execute the `run_app.bat` file in your terminal or double-click it in the File Explorer. This will open the Streamlit application in your default web browser.

## Screenshots / Demo
*(Include screenshots or GIFs of the Streamlit application here)*
