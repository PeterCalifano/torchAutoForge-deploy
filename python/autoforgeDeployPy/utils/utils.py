import os

def get_file_size(file_path : str) -> float:
    size = os.path.getsize(file_path)
    return float(size)
