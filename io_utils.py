# io_utils.py
import glob
import json
import numpy as np

def list_images(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))

def save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def ndarray_to_list(x):
    return x.tolist() if isinstance(x, np.ndarray) else x
