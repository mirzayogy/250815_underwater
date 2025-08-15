from pathlib import Path
def ensure_folder_exists(folder_path):
    path = Path(folder_path)
    path.mkdir(parents=True, exist_ok=True)  # parents=True => buat folder di atasnya kalau belum ada
    # print(f"Folder siap digunakan: {path.resolve()}")