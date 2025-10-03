# zhang.py
import numpy as np
import math
from dataclasses import dataclass

@dataclass
class Intrinsics:
    K: np.ndarray
    dist: np.ndarray | None

def _vec_v(H, i, j):
    h = H.T
    hi, hj = h[i - 1], h[j - 1]
    return np.array([
        hi[0]*hj[0],
        hi[0]*hj[1] + hi[1]*hj[0],
        hi[1]*hj[1],
        hi[2]*hj[0] + hi[0]*hj[2],
        hi[2]*hj[1] + hi[1]*hj[2],
        hi[2]*hj[2],
    ])

def compute_B_from_homographies(Hs):
    V = []
    for H in Hs:
        V.append(_vec_v(H, 1, 2))
        V.append(_vec_v(H, 1, 1) - _vec_v(H, 2, 2))
    V = np.vstack(V)
    _, _, VT = np.linalg.svd(V)
    b = VT[-1]
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])
    return B

def K_from_B(B):
    B11,B12,B22 = B[0,0],B[0,1],B[1,1]
    B13,B23,B33 = B[0,2],B[1,2],B[2,2]
    v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
    lam = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
    alpha = math.sqrt(lam / B11)
    beta  = math.sqrt(lam * B11 / (B11*B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lam
    u0 = (gamma * v0 / alpha) - (B13 * alpha**2 / lam)
    return np.array([[alpha, gamma, u0],
                     [0.0,   beta,  v0],
                     [0.0,   0.0,   1.0]])

def homography_DLT(src_pts, dst_pts):
    def normalize(pts):
        mean = pts.mean(axis=0)
        std = np.sqrt(((pts-mean)**2).sum(axis=1).mean()/2.0)
        s = np.sqrt(2)/std
        T = np.array([[s,0,-s*mean[0]],[0,s,-s*mean[1]],[0,0,1]])
        pts_h = np.c_[pts, np.ones(len(pts))].T
        n = (T @ pts_h).T
        return n[:, :2], T
    src_n, Ts = normalize(src_pts)
    dst_n, Td = normalize(dst_pts)

    A = []
    for (X,Y), (x,y) in zip(src_n, dst_n):
        A.append([0,0,0, -X,-Y,-1, y*X, y*Y, y])
        A.append([X,Y,1,  0, 0, 0,-x*X,-x*Y,-x])
    A = np.asarray(A)
    _,_,VT = np.linalg.svd(A)
    h = VT[-1].reshape(3,3)
    H = np.linalg.inv(Td) @ h @ Ts
    return H / H[2,2]

def extrinsics_from_H(K, H):
    Kinv = np.linalg.inv(K)
    h1,h2,h3 = H[:,0], H[:,1], H[:,2]
    lam = 1.0/np.linalg.norm(Kinv @ h1)
    r1 = lam*(Kinv@h1)
    r2 = lam*(Kinv@h2)
    r3 = np.cross(r1, r2)
    R = np.c_[r1,r2,r3]
    U,_,VT = np.linalg.svd(R)
    R = U @ VT
    t = lam*(Kinv@h3)
    return R, t

def zhang_calibration(obj_pts, img_pts):
    Hs = []
    for obj, img in zip(obj_pts, img_pts):
        XY = obj[:, :2]
        H = homography_DLT(XY, img)
        Hs.append(H)
    B = compute_B_from_homographies(Hs)
    K = K_from_B(B)

    extr = []
    reproj = []
    errs = []
    for (obj, img), H in zip(zip(obj_pts, img_pts), Hs):
        R,t = extrinsics_from_H(K, H)
        extr.append((R,t))

        Rt = np.c_[R, t]
        XYZ1 = np.c_[obj[:, :2], np.zeros(len(obj)), np.ones(len(obj))].T
        proj = K @ (Rt @ XYZ1)
        proj = (proj[:2]/proj[2]).T
        reproj.append(proj)
        errs.extend(np.linalg.norm(proj - img, axis=1))

    mean_err = float(np.mean(errs))
    return Intrinsics(K=K, dist=None), extr, reproj, mean_err
