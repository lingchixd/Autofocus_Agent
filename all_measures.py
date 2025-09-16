# traditional method

"""
focus_metrics_all.py
- Scan level folders (0..20 by default, or auto-detect numeric subfolders)
- For each level, take FIRST_N files (sorted), compute multiple focus measures
- Print results to console

Dependencies:
  numpy, imageio, scipy (ndimage, signal, fft), pywt
Install (if needed):
  pip install numpy imageio scipy pywavelets
"""

import os, glob, math
import numpy as np
import imageio.v3 as iio
from typing import List, Tuple

from scipy.ndimage import sobel, laplace
from scipy.signal import convolve2d
from scipy.fft import dctn
import pywt

import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
BASE_DIR = r"E:\SS316_ShotPeen\images"  # change to your root
AUTO_DETECT_LEVELS = True               # if True, auto-detect numeric subfolders
DEFAULT_LEVELS = list(range(0, 21))     # fallback: 0..20
TAKE_FIRST_N = 1                        # set 50 if you want the first 50 images per level

SUPPORTED_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")

# ---------------------------
# Utils
# ---------------------------
def list_level_dirs(base: str) -> List[Tuple[int, str]]:
    """Return [(level_id, folder_path), ...] sorted by level_id."""
    if AUTO_DETECT_LEVELS:
        pairs = []
        for name in os.listdir(base):
            full = os.path.join(base, name)
            if os.path.isdir(full) and name.isdigit():
                pairs.append((int(name), full))
        return sorted(pairs, key=lambda x: x[0])
    else:
        return [(lv, os.path.join(base, str(lv))) for lv in DEFAULT_LEVELS]

def list_images(folder: str) -> List[str]:
    files = []
    for pat in SUPPORTED_EXTS:
        files.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(files)

def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert to 2D grayscale float64 (drop alpha if present)."""
    if img.ndim == 2:
        return img.astype(np.float64, copy=False)
    if img.ndim == 3:
        c = img.shape[-1]
        if c == 1:
            return img[..., 0].astype(np.float64, copy=False)
        # ITU-R BT.601 luma; ignore alpha if present
        r = img[..., 0].astype(np.float64, copy=False)
        g = img[..., 1].astype(np.float64, copy=False)
        b = img[..., 2].astype(np.float64, copy=False)
        return 0.299 * r + 0.587 * g + 0.114 * b
    raise ValueError(f"Unsupported image ndim={img.ndim}")

def mean2(a: np.ndarray) -> float:
    return float(np.mean(a))

def std2(a: np.ndarray) -> float:
    return float(np.std(a))

def abslog2(x: float) -> float:
    if x > 0:
        return math.log2(x)
    elif x < 0:
        return math.log2(-x)
    else:
        return 0.0

# Laplacian kernel similar to MATLAB fspecial('laplacian', alpha=0.2)
def laplacian_kernel_fspecial() -> np.ndarray:
    return np.array([[0.1667, 0.6667, 0.1667],
                     [0.6667, -3.3333, 0.6667],
                     [0.1667, 0.6667, 0.1667]], dtype=np.float64)


# Focus measures

def BREN(img: np.ndarray) -> float:
    """Brenner gradient (two-pixel difference), mean of max(DH,DV)^2."""
    M, N = img.shape
    DH = np.zeros((M, N), dtype=np.float64)
    DV = np.zeros((M, N), dtype=np.float64)
    DV[:M-2, :] = img[2:, :] - img[:M-2, :]
    DH[:, :N-2] = img[:, 2:] - img[:, :N-2]
    FM = np.maximum(DH, DV)
    FM = FM * FM
    return mean2(FM)

def LAPE(img: np.ndarray) -> float:
    """Mean of squared Laplacian response using fspecial-like kernel."""
    K = laplacian_kernel_fspecial()
    FM = convolve2d(img, K, mode='same', boundary='symm')
    return mean2(FM * FM)

def LAPM(img: np.ndarray) -> float:
    """Mean of |Lx|+|Ly| with 1D second-derivative kernel [-1,2,-1]."""
    M = np.array([-1.0, 2.0, -1.0], dtype=np.float64)
    Lx = convolve2d(img, M[np.newaxis, :], mode='same', boundary='symm')
    Ly = convolve2d(img, M[:, np.newaxis], mode='same', boundary='symm')
    FM = np.abs(Lx) + np.abs(Ly)
    return mean2(FM)

def LAPV(img: np.ndarray) -> float:
    """Variance of Laplacian using fspecial-like kernel (std2(ILAP)^2)."""
    K = laplacian_kernel_fspecial()
    ILAP = convolve2d(img, K, mode='same', boundary='symm')
    s = std2(ILAP)
    return s * s

def LAPD(img: np.ndarray) -> float:
    """Sum of absolute responses of 4 directional Laplacian-like filters."""
    M1 = np.array([-1.0, 2.0, -1.0], dtype=np.float64)
    M2 = np.array([[0, 0, -1],
                   [0, 2,  0],
                   [-1, 0, 0]], dtype=np.float64) / math.sqrt(2.0)
    M3 = np.array([[-1, 0, 0],
                   [ 0, 2, 0],
                   [ 0, 0,-1]], dtype=np.float64) / math.sqrt(2.0)

    F1 = convolve2d(img, M1[np.newaxis, :], mode='same', boundary='symm')
    F4 = convolve2d(img, M1[:, np.newaxis], mode='same', boundary='symm')
    F2 = convolve2d(img, M2, mode='same', boundary='symm')
    F3 = convolve2d(img, M3, mode='same', boundary='symm')
    FM = np.abs(F1) + np.abs(F2) + np.abs(F3) + np.abs(F4)
    return mean2(FM)

def VOLA(img: np.ndarray) -> float:
    """Vollath's F4-like measure."""
    Image = img.astype(np.float64, copy=False)
    I1 = Image.copy(); I1[:-1, :] = Image[1:, :]
    I2 = Image.copy(); I2[:-2, :] = Image[2:, :]
    X = Image * (I1 - I2)
    return mean2(X)

def TENV(img: np.ndarray) -> float:
    """Variance of Sobel gradient magnitude (std2(G)^2)."""
    Gx = sobel(img, axis=1, mode='reflect')
    Gy = sobel(img, axis=0, mode='reflect')
    G = Gx * Gx + Gy * Gy
    s = std2(G)
    return s * s

def TENG(img: np.ndarray) -> float:
    """Mean of Sobel gradient magnitude squared."""
    Gx = sobel(img, axis=1, mode='reflect')
    Gy = sobel(img, axis=0, mode='reflect')
    FM = Gx * Gx + Gy * Gy
    return mean2(FM)

def WAVS(img: np.ndarray) -> float:
    """Mean of |H|+|V|+|D| from 1-level DWT (db6)."""
    coeffs2 = pywt.wavedec2(img, wavelet='db6', level=1, mode='symmetric')
    cA, (cH, cV, cD) = coeffs2
    FM = np.abs(cH) + np.abs(cV) + np.abs(cD)
    return mean2(FM)

def WAVV(img: np.ndarray) -> float:
    """Variance-based measure from |H|,|V|,|D| (std2(H)^2 + std2(V) + std2(D))."""
    coeffs2 = pywt.wavedec2(img, wavelet='db6', level=1, mode='symmetric')
    cA, (cH, cV, cD) = coeffs2
    return std2(np.abs(cH))**2 + std2(np.abs(cV)) + std2(np.abs(cD))

def DCTS(img: np.ndarray, r: int = 100) -> float:
    """DCT spectral entropy variant in top-left r×r disk."""
    # Optional prefilter as in MATLAB: imgaussfilt(img) — here we skip or add if needed.
    C = dctn(img, type=2, norm='ortho')
    H, W = C.shape
    r = min(r, H, W)

    value = 0.0
    L2 = float(np.linalg.norm(C))
    if L2 == 0:
        return 0.0
    for i in range(1, r + 1):
        for j in range(1, r + 1):
            if i*i + j*j < r*r:
                val = C[i-1, j-1] / L2
                # MATLAB-style abslog2(val)
                value += abs(val) * abslog2(val)
    fm = value * -2.0 / (r * r)
    return float(fm)

def HISE(img: np.ndarray, bins: int = 256) -> float:
    """Entropy of grayscale histogram (bits)."""
    # mimic MATLAB entropy(img) roughly
    arr = img.astype(np.float64, copy=False)
    # normalize to [0,1] if range is larger
    a_min, a_max = np.min(arr), np.max(arr)
    if a_max > a_min:
        arr = (arr - a_min) / (a_max - a_min)
    hist, _ = np.histogram(arr, bins=bins, range=(0.0, 1.0), density=False)
    p = hist.astype(np.float64)
    s = p.sum()
    if s == 0:
        return 0.0
    p /= s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def SFIL(img: np.ndarray, wsize: int = 15) -> float:
    """Steerable-like: first-order Gaussian derivatives in 8 orientations, take max; mean2."""
    N = wsize // 2
    sigma = N / 2.5 if N > 0 else 1.0
    y, x = np.mgrid[-N:N+1, -N:N+1]
    G = np.exp(-(x*x + y*y) / (2.0 * sigma * sigma)) / (2.0 * math.pi * sigma)
    # first derivatives of Gaussian
    Gx = (-x * G) / (sigma * sigma)
    Gy = (-y * G) / (sigma * sigma)
    # normalize to match MATLAB's per-kernel sum normalization intent
    sx = np.sum(np.abs(Gx)); sy = np.sum(np.abs(Gy))
    if sx > 0: Gx = Gx / sx
    if sy > 0: Gy = Gy / sy

    # responses at 8 orientations (0,45,90,...,315)
    R = []
    Fx = convolve2d(img, Gx, mode='same', boundary='symm')
    Fy = convolve2d(img, Gy, mode='same', boundary='symm')
    # base components
    R0   =  math.cos(math.radians(  0))*Fx + math.sin(math.radians(  0))*Fy
    R45  =  math.cos(math.radians( 45))*Fx + math.sin(math.radians( 45))*Fy
    R90  =  math.cos(math.radians( 90))*Fx + math.sin(math.radians( 90))*Fy
    R135 =  math.cos(math.radians(135))*Fx + math.sin(math.radians(135))*Fy
    R180 =  math.cos(math.radians(180))*Fx + math.sin(math.radians(180))*Fy
    R225 =  math.cos(math.radians(225))*Fx + math.sin(math.radians(225))*Fy
    R270 =  math.cos(math.radians(270))*Fx + math.sin(math.radians(270))*Fy
    R315 =  math.cos(math.radians(315))*Fx + math.sin(math.radians(315))*Fy

    # stack and take per-pixel max across orientations
    Rstack = np.stack([R0, R45, R90, R135, R180, R225, R270, R315], axis=0)
    FM = np.max(Rstack, axis=0)
    return mean2(FM)


# Runner

def main():
    print(f"[INFO] Base: {BASE_DIR}")
    levels = list_level_dirs(BASE_DIR)
    if not levels:
        print("[ERROR] No level folders found.")
        return
    print(f"[INFO] Detected levels: {[lv for lv, _ in levels]}")
    print(f"[INFO] TAKE_FIRST_N per level: {TAKE_FIRST_N}")
    print("-"*80)

    total = 0
    for lv, folder in levels:
        files = list_images(folder)
        if not files:
            print(f"[WARN] Level {lv}: no images.")
            print("-"*80); continue

        print(f"[LEVEL {lv}] Folder: {folder}")
        cnt = 0
        for fp in files[:TAKE_FIRST_N]:
            try:
                img = iio.imread(fp)
                g = to_gray(img)

                # compute all metrics
                fm_vals = {
                    "BREN": BREN(g),
                    "LAPE": LAPE(g),
                    "LAPM": LAPM(g),
                    "LAPV": LAPV(g),
                    "LAPD": LAPD(g),
                    "VOLA": VOLA(g),
                    "TENV": TENV(g),
                    "TENG": TENG(g),
                    "WAVS": WAVS(g),
                    "WAVV": WAVV(g),
                    "DCTS": DCTS(g, r=100),
                    "HISE": HISE(g),
                    "SFIL": SFIL(g, wsize=15),
                }

                print(f"  • {os.path.basename(fp)}")
                # concise one-line summary
                summary = " | ".join(f"{k}={v:.6g}" for k, v in fm_vals.items())
                print(f"    {summary}")

                cnt += 1
                total += 1
            except Exception as e:
                print(f"  ! Error processing {fp}: {e}")

        print(f"  -> processed {cnt} image(s) in level {lv}")
        print("-"*80)

    print(f"[SUMMARY] Total images processed: {total}")

def main_plot():
    levels = []
    if AUTO_DETECT_LEVELS:
        for name in os.listdir(BASE_DIR):
            full = os.path.join(BASE_DIR, name)
            if os.path.isdir(full) and name.isdigit():
                levels.append((int(name), full))
        levels.sort(key=lambda x: x[0])
    else:
        levels = [(lv, os.path.join(BASE_DIR, str(lv))) for lv in DEFAULT_LEVELS]

    if not levels:
        print("[ERROR] No level folders found."); return

    metrics = ["BREN","LAPE","LAPM","LAPV","LAPD",
               "VOLA","TENV","TENG","WAVS","WAVV","DCTS","HISE"]

    # prepare plot for each parameter
    scores_all = {m: [] for m in metrics}
    levels_x = []

    for lv, folder in levels:
        files = []
        for pat in SUPPORTED_EXTS:
            files.extend(glob.glob(os.path.join(folder, pat)))
        files = sorted(files)
        if not files:
            # make x-axis continuous
            for m in metrics: scores_all[m].append(float("nan"))
            levels_x.append(lv)
            continue

        files = files[:TAKE_FIRST_N]
        # average
        acc = {m: [] for m in metrics}

        for fp in files:
            try:
                img = iio.imread(fp)
                g = to_gray(img)

                acc["BREN"].append(BREN(g))
                acc["LAPE"].append(LAPE(g))
                acc["LAPM"].append(LAPM(g))
                acc["LAPV"].append(LAPV(g))
                acc["LAPD"].append(LAPD(g))
                acc["VOLA"].append(VOLA(g))
                acc["TENV"].append(TENV(g))
                acc["TENG"].append(TENG(g))
                acc["WAVS"].append(WAVS(g))
                acc["WAVV"].append(WAVV(g))
                acc["DCTS"].append(DCTS(g, r=100))
                acc["HISE"].append(HISE(g))
            except Exception as e:
                print(f"[WARN] {fp} failed: {e}")

        # average in different level
        for m in metrics:
            vals = np.array(acc[m], dtype=float)
            scores_all[m].append(float(np.nanmean(vals)) if vals.size else float("nan"))

        levels_x.append(lv)

    # sub-image
    plt.figure(figsize=(16, 12))
    for idx, m in enumerate(metrics, 1):
        plt.subplot(4, 3, idx)
        plt.plot(levels_x, scores_all[m], marker='o')
        plt.title(m)
        plt.xlabel("Level")
        plt.ylabel("Score")
        plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    main_plot()

