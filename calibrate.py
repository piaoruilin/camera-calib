# calibrate.py
import os
import cv2
import numpy as np

from config import IMAGES_GLOB, PATTERN_SIZE, SQUARE_SIZE_MM, SUBPIX_WINDOW
from io_utils import list_images, save_json, ndarray_to_list
from corner_detect import find_corners, make_object_points
from zhang import zhang_calibration
from opencv_calib import opencv_calibration
from visualize import save_overlay

def main():
    image_paths = list_images(IMAGES_GLOB)
    print(f"Found {len(image_paths)} images in {IMAGES_GLOB}")
    if not image_paths:
        raise SystemExit("No images found. Put your photos under data/")

    obj_pts, img_pts, image_sizes, gray_imgs, good_paths = find_corners(
        image_paths, PATTERN_SIZE, SUBPIX_WINDOW
    )
    if not img_pts:
        raise SystemExit(
            "No corners detected. If your board is 10x7 squares, PATTERN_SIZE must be (9,6)."
        )

    # Build the repeated object points
    objp = make_object_points(PATTERN_SIZE, SQUARE_SIZE_MM)
    obj_pts = [objp.copy() for _ in img_pts]

    assert len(set(image_sizes)) == 1, "All images must have the same size."
    W, H = image_sizes[0]

    # --- Zhang (no distortion) ---
    zK, z_ext, z_reprojs, z_err = zhang_calibration(obj_pts, img_pts)
    print("\nZhang intrinsics:\n", zK.K)
    print("Zhang mean reprojection error (px):", z_err)

    # --- OpenCV (with distortion) ---
    oK, o_ext, o_reprojs, o_err = opencv_calibration(obj_pts, img_pts, (W, H))
    print("\nOpenCV intrinsics:\n", oK.K)
    print("OpenCV distortion:", oK.dist.ravel())
    print("OpenCV mean reprojection error (px):", o_err)

    # --- Visualizations & sample undistort ---
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    for i, (g, det, zrp, orp) in enumerate(zip(gray_imgs, img_pts, z_reprojs, o_reprojs)):
        save_overlay(os.path.join(out_dir, f"reproj_zhang_{i:02d}.png"), g, det, zrp, "Zhang reproj")
        save_overlay(os.path.join(out_dir, f"reproj_opencv_{i:02d}.png"), g, det, orp, "OpenCV reproj")

    sample = cv2.imread(good_paths[0])
    und = cv2.undistort(sample, oK.K, oK.dist)
    cv2.imwrite(os.path.join(out_dir, "undistorted_sample.png"), und)

    # --- JSON summaries ---
    z_json = {
        "K": ndarray_to_list(zK.K),
        "mean_reprojection_error_px": z_err,
        "per_view": [{"R": ndarray_to_list(R), "t": ndarray_to_list(t)} for (R, t) in z_ext],
    }
    o_json = {
        "K": ndarray_to_list(oK.K),
        "distortion": ndarray_to_list(oK.dist.ravel()),
        "mean_reprojection_error_px": o_err,
        "per_view": [{"R": ndarray_to_list(R), "t": ndarray_to_list(t)} for (R, t) in o_ext],
    }
    save_json(os.path.join(out_dir, "results_zhang.json"), z_json)
    save_json(os.path.join(out_dir, "results_opencv.json"), o_json)
    print("\nSaved outputs in ./out/")

if __name__ == "__main__":
    main()
