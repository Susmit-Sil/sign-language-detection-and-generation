import os
import sys
import zipfile
from huggingface_hub import HfApi, login

EXCLUDE_DIRS = {'.git', 'venv', 'signlang_venv', '__pycache__', '.gemini', '.agents'}

def zip_folder(folder_path, zip_path):
    print(f"Creating zip archive: {zip_path} from {folder_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        for root, dirs, files in os.walk(folder_path):
            # Exclude folders
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                zipf.write(file_path, arcname)
    print(f"Zip created successfully: {zip_path} ({round(os.path.getsize(zip_path) / (1024*1024), 2)} MB)")

def main():
    if len(sys.argv) < 3:
        print("Usage: python zip_and_upload.py <token> <repo_name>")
        sys.exit(1)
        
    token = sys.argv[1]
    repo_name = sys.argv[2]
    
    print("Logging into Hugging Face...")
    try:
        login(token=token)
    except Exception as e:
        print(f"Login failed: {e}")
        sys.exit(1)
        
    api = HfApi()
    username = api.whoami()['name']
    repo_id = f"{username}/{repo_name}"
    
    print(f"Ensuring dataset repository '{repo_id}' exists...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    
    # 1. Zip asl_datasets
    asl_zip = "asl_datasets.zip"
    if os.path.exists("asl_datasets") and not os.path.exists(asl_zip):
        zip_folder("asl_datasets", asl_zip)
    elif os.path.exists(asl_zip):
        print(f"Using existing zip: {asl_zip}")
        
    # 2. Zip RealTimeObjectDetection (excluding signlang_venv)
    rtod_zip = "RealTimeObjectDetection.zip"
    if os.path.exists("RealTimeObjectDetection") and not os.path.exists(rtod_zip):
        zip_folder("RealTimeObjectDetection", rtod_zip)
    elif os.path.exists(rtod_zip):
        print(f"Using existing zip: {rtod_zip}")
        
    # Upload the zip files
    for zip_file in [asl_zip, rtod_zip]:
        if os.path.exists(zip_file):
            print(f"\n[START] Uploading '{zip_file}' to '{repo_id}'...")
            try:
                api.upload_file(
                    path_or_fileobj=zip_file,
                    path_in_repo=zip_file,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                print(f"[SUCCESS] Uploaded '{zip_file}' successfully!")
            except Exception as e:
                print(f"[ERROR] Failed to upload '{zip_file}': {e}")
                
    print("\n[FINISHED] All zip files uploaded!")

if __name__ == "__main__":
    main()
