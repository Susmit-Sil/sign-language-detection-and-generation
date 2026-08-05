import sys
import os
from huggingface_hub import HfApi, login

def main():
    if len(sys.argv) < 3:
        print("Usage: python upload_bg.py <token> <repo_name>")
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
    
    ignore_patterns = ["**/venv/**", "**/signlang_venv/**", "**/.git/**", "**/__pycache__/**"]
    
    # Folders to upload
    targets = []
    if os.path.exists("asl_datasets"):
        targets.append(("asl_datasets", "asl_datasets"))
    if os.path.exists("RealTimeObjectDetection"):
        targets.append(("RealTimeObjectDetection", "RealTimeObjectDetection"))
        
    for name, path in targets:
        print(f"\n[START] Uploading '{name}' to '{repo_id}'...")
        try:
            api.upload_folder(
                folder_path=path,
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=name,
                ignore_patterns=ignore_patterns,
            )
            print(f"[SUCCESS] Uploaded '{name}' successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to upload '{name}': {e}")
            
    print("\n[FINISHED] All uploads completed!")

if __name__ == "__main__":
    main()
