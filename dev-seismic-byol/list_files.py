import os

def count_files_in_folders(path):
    try:
        for root, dirs, files in os.walk(path):
            folder_name = os.path.basename(root)
            print(f"Folder: {folder_name or '/'} - Files: {len(files)}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    
    root = 'ckpt/train'
    # root = 'logs/test'
    
    for i in range(len(os.listdir(root))):
        path = f"{root}/{i}"
        lst = os.listdir(path)        
        print(f'{path}: {len(lst)}')