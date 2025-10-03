# visualize.py
import cv2
import numpy as np
import os

def save_overlay(path, gray, detected, reproj, title):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # detected: yellow
    for pt in detected:
        cv2.circle(vis, tuple(np.int32(pt)), 3, (0,255,255), -1)
    # reprojected: magenta
    for pt in reproj:
        cv2.circle(vis, tuple(np.int32(pt)), 2, (255,0,255), -1)
    cv2.putText(vis, title, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10,200,10), 2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, vis)
