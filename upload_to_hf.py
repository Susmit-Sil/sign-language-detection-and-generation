import os
import sys
from huggingface_hub import HfApi, login

# Folders to exclude completely from size calculation and uploads
EXCLUDE_DIRS = {'.git', 'venv', 'signlang_venv', '__pycache__', '.gemini', '.agents'}

def get_folder_size_mb(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        # Modify dirnames in-place to prevent walking into excluded directories
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    pass
    return round(total_size / (1024 * 1024), 2)

def upload_dataset():
    print("=" * 60)
    print("      Hugging Face Dataset Upload Helper")
    print("=" * 60)
    
    # 1. Get Token
    token = input("Please enter your Hugging Face WRITE token (get one at huggingface.co/settings/tokens): ").strip()
    if not token:
        print("Error: Token is required.")
        return
        
    try:
        login(token=token)
        print("Successfully logged in to Hugging Face!")
    except Exception as e:
        print(f"Login failed: {e}")
        return

    # 2. Get Repository Details
    api = HfApi()
    user_info = api.whoami()
    username = user_info['name']
    print(f"Logged in as user: {username}")
    
    repo_name = input("Enter the repository name (e.g. sign-language-dataset): ").strip()
    if not repo_name:
        print("Error: Repository name is required.")
        return
        
    repo_id = f"{username}/{repo_name}"
    
    # 3. Create Dataset Repo if it doesn't exist
    print(f"Ensuring dataset repository '{repo_id}' exists...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"Repository {repo_id} is ready.")
    except Exception as e:
        print(f"Error checking/creating repository: {e}")
        return

    # 4. Scan for uploadable folders
    print("\nScanning workspace for folders...")
    
    uploadable = []
    
    # Scan root directory
    for item in os.listdir('.'):
        if os.path.isdir(item) and item not in EXCLUDE_DIRS:
            size = get_folder_size_mb(item)
            uploadable.append((item, os.path.abspath(item), size))
            
    # Also scan RealTimeObjectDetection subfolders (excluding signlang_venv)
    rtod_dir = 'RealTimeObjectDetection'
    if os.path.isdir(rtod_dir):
        for item in os.listdir(rtod_dir):
            sub_path = os.path.join(rtod_dir, item)
            if os.path.isdir(sub_path) and item not in EXCLUDE_DIRS:
                size = get_folder_size_mb(sub_path)
                uploadable.append((f"{rtod_dir}/{item}", os.path.abspath(sub_path), size))

    if not uploadable:
        print("No folders found for upload.")
        return

    # Sort by size descending
    uploadable.sort(key=lambda x: x[2], reverse=True)

    print("\n--- Available folders to upload (Excluding virtual environments) ---")
    for idx, (name, path, size) in enumerate(uploadable, 1):
        print(f" [{idx}] {name:<40} Size: {size:>8} MB")
        
    print("\nNOTE: Virtual environments ('venv' and 'signlang_venv') are completely excluded.")

    choice = input("\nEnter the number of the folder to upload, 'all' to upload all listed, or 'custom' to enter a path: ").strip().lower()
    
    selected = []
    if choice == 'all':
        selected = [(name, path) for name, path, _ in uploadable]
    elif choice == 'custom':
        custom_path = input("Enter absolute or relative path to folder/file: ").strip()
        if not os.path.exists(custom_path):
            print("Path does not exist.")
            return
        selected = [(os.path.basename(custom_path), os.path.abspath(custom_path))]
    else:
        try:
            idx = int(choice) - 1 if choice else 0
            if 0 <= idx < len(uploadable):
                selected = [(uploadable[idx][0], uploadable[idx][1])]
            else:
                print("Invalid selection.")
                return
        except ValueError:
            print("Invalid selection.")
            return

    # Ignore patterns list to prevent uploading environment directories if chosen manually
    ignore_patterns = ["**/venv/**", "**/signlang_venv/**", "**/.git/**", "**/__pycache__/**"]

    for name, path in selected:
        size = get_folder_size_mb(path)
        print(f"\nUploading '{name}' ({size} MB) to Hugging Face '{repo_id}'...")
        try:
            if os.path.isdir(path):
                api.upload_folder(
                    folder_path=path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    path_in_repo=name,
                    ignore_patterns=ignore_patterns,
                )
                print(f"Successfully uploaded folder: {name}!")
            else:
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=os.path.basename(path),
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                print(f"Successfully uploaded file: {os.path.basename(path)}!")
        except Exception as e:
            print(f"Failed to upload {name}: {e}")
            
    print("\n" + "=" * 60)
    print("Upload process finished!")
    print(f"View your dataset at: https://huggingface.co/datasets/{repo_id}")
    print("=" * 60)

if __name__ == "__main__":
    try:
        upload_dataset()
    except KeyboardInterrupt:
        print("\nUpload canceled by user.")
        sys.exit(0)
