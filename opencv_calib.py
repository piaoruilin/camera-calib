# opencv_calib.py
import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class Intrinsics:
    K: np.ndarray
    dist: np.ndarray | None

def opencv_calibration(obj_pts, img_pts, image_size):
    obj = [op.reshape(-1,1,3) for op in obj_pts]
    img = [ip.reshape(-1,1,2) for ip in img_pts]

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, image_size, None, None)
    extr = []
    reproj = []
    errs = []
    for op, ip, rvec, tvec in zip(obj_pts, img_pts, rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        extr.append((R,t))
        proj, _ = cv2.projectPoints(op, rvec, tvec, K, dist)
        proj = proj.reshape(-1,2)
        reproj.append(proj)
        errs.extend(np.linalg.norm(proj - ip, axis=1))
    return Intrinsics(K=K, dist=dist), extr, reproj, float(np.mean(errs))
