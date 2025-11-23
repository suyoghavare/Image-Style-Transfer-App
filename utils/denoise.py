import cv2
import numpy as np


def bilateral_denoise(frame_bgr,
                      d: int = 9,
                      sigma_color: float = 75,
                      sigma_space: float = 75):
    return cv2.bilateralFilter(frame_bgr, d, sigma_color, sigma_space)


def anisotropic_diffusion_gray(img,
                               n_iter: int = 10,
                               k: float = 15,
                               lam: float = 0.25):

    img = img.astype(np.float32)
    out = img.copy()

    for _ in range(n_iter):
        north = np.roll(out, -1, axis=0)
        south = np.roll(out, 1, axis=0)
        east = np.roll(out, -1, axis=1)
        west = np.roll(out, 1, axis=1)

        dN = north - out
        dS = south - out
        dE = east - out
        dW = west - out

        cN = np.exp(-(dN / k) ** 2)
        cS = np.exp(-(dS / k) ** 2)
        cE = np.exp(-(dE / k) ** 2)
        cW = np.exp(-(dW / k) ** 2)

        out = out + lam * (cN * dN + cS * dS + cE * dE + cW * dW)

    return np.clip(out, 0.0, 1.0)
