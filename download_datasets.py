"""
download_datasets.py
────────────────────
Helper script to download ASL datasets from Kaggle and other sources.

Run this once to set up your datasets directory.
"""

import os
import subprocess
import sys
from pathlib import Path

HACKATHON_DIR = Path(__file__).parent
DATASETS_DIR = HACKATHON_DIR / "asl_datasets"

def check_kaggle_setup():
    """Check if Kaggle CLI is installed and configured."""
    try:
        import kaggle
        print("✅ Kaggle CLI is installed")
        return True
    except ImportError:
        print("❌ Kaggle CLI not installed")
        print("   Run: pip install kaggle")
        return False

def download_asl_alphabet():
    """Download ASL Alphabet dataset from Kaggle."""
    print("\n📥 Downloading ASL Alphabet dataset...")
    dataset_path = DATASETS_DIR / "asl_alphabet"
    
    if dataset_path.exists():
        print(f"⚠️  Dataset already exists at {dataset_path}")
        response = input("   Re-download? (y/n): ")
        if response.lower() != 'y':
            return
    
    os.makedirs(dataset_path, exist_ok=True)
    
    try:
        cmd = [
            "kaggle", "datasets", "download", 
            "-d", "grassknoted/asl-alphabet",
            "-p", str(dataset_path),
            "--unzip"
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ ASL Alphabet downloaded to {dataset_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download: {e}")
        print("\n💡 Manual download:")
        print("   1. Go to: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print("   2. Download ZIP")
        print(f"   3. Extract to: {dataset_path}")

def download_sign_mnist():
    """Download Sign Language MNIST from Kaggle."""
    print("\n📥 Downloading Sign Language MNIST...")
    dataset_path = DATASETS_DIR / "sign_mnist"
    
    if dataset_path.exists():
        print(f"⚠️  Dataset already exists at {dataset_path}")
        response = input("   Re-download? (y/n): ")
        if response.lower() != 'y':
            return
    
    os.makedirs(dataset_path, exist_ok=True)
    
    try:
        cmd = [
            "kaggle", "datasets", "download",
            "-d", "datamunge/sign-language-mnist",
            "-p", str(dataset_path),
            "--unzip"
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ Sign MNIST downloaded to {dataset_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download: {e}")
        print("\n💡 Manual download:")
        print("   1. Go to: https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
        print("   2. Download ZIP")
        print(f"   3. Extract to: {dataset_path}")

def download_ayuraj_asl():
    """Download Ayuraj ASL dataset from Kaggle."""
    print("\n📥 Downloading Ayuraj ASL dataset...")
    dataset_path = DATASETS_DIR / "ayuraj_asl"
    
    if dataset_path.exists():
        print(f"⚠️  Dataset already exists at {dataset_path}")
        response = input("   Re-download? (y/n): ")
        if response.lower() != 'y':
            return
    
    os.makedirs(dataset_path, exist_ok=True)
    
    try:
        cmd = [
            "kaggle", "datasets", "download",
            "-d", "ayuraj/asl-dataset",
            "-p", str(dataset_path),
            "--unzip"
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ Ayuraj ASL downloaded to {dataset_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download: {e}")
        print("\n💡 Manual download:")
        print("   1. Go to: https://www.kaggle.com/datasets/ayuraj/asl-dataset")
        print("   2. Download ZIP")
        print(f"   3. Extract to: {dataset_path}")

def main():
    print("=" * 70)
    print("ASL Datasets Downloader")
    print("=" * 70)
    
    # Create datasets directory
    os.makedirs(DATASETS_DIR, exist_ok=True)
    print(f"\n📁 Datasets will be saved to: {DATASETS_DIR}")
    
    # Check Kaggle setup
    if not check_kaggle_setup():
        print("\n⚠️  Please install Kaggle CLI first:")
        print("   pip install kaggle")
        print("\n   Then configure your API credentials:")
        print("   1. Go to https://www.kaggle.com/settings")
        print("   2. Create New API Token")
        print("   3. Place kaggle.json in ~/.kaggle/ (or C:\\Users\\<you>\\.kaggle\\)")
        sys.exit(1)
    
    # Download datasets
    print("\n" + "=" * 70)
    print("Which datasets would you like to download?")
    print("=" * 70)
    print("1. ASL Alphabet (3GB, high quality color images)")
    print("2. Sign Language MNIST (small, grayscale 28x28)")
    print("3. Ayuraj ASL Dataset (varied, good for testing)")
    print("4. All of them")
    print("0. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        download_asl_alphabet()
    elif choice == '2':
        download_sign_mnist()
    elif choice == '3':
        download_ayuraj_asl()
    elif choice == '4':
        download_asl_alphabet()
        download_sign_mnist()
        download_ayuraj_asl()
    elif choice == '0':
        print("Exiting...")
        return
    else:
        print("Invalid choice")
        return
    
    print("\n" + "=" * 70)
    print("✅ Download complete!")
    print("=" * 70)
    print(f"\nDatasets location: {DATASETS_DIR}")
    print("\nNext steps:")
    print("1. Run the Streamlit app: streamlit run app.py")
    print("2. The app will now use real ASL images!")

if __name__ == "__main__":
    main()
