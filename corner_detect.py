# corner_detect.py
import cv2
import numpy as np
from typing import List, Tuple

def make_object_points(pattern_size: tuple[int, int], square_size_mm: float) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * float(square_size_mm)  # units: mm
    return objp

def find_corners(
    image_paths: list[str],
    pattern_size: tuple[int, int],
    subpix_window: tuple[int, int],
):
    obj_pts: List[np.ndarray] = []
    img_pts: List[np.ndarray] = []
    image_sizes: List[Tuple[int, int]] = []
    gray_imgs: List[np.ndarray] = []
    good_paths: List[str] = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[warn] could not read: {p}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            print(f"[warn] corners NOT found: {p}")
            continue

        corners = cv2.cornerSubPix(gray, corners, subpix_window, (-1, -1), criteria)
        img_pts.append(corners.reshape(-1, 2))
        gray_imgs.append(gray)
        image_sizes.append(gray.shape[::-1])  # (w, h)
        good_paths.append(p)

    return obj_pts, img_pts, image_sizes, gray_imgs, good_paths
