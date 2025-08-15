import math
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os

# ---------------------------
# Utils
# ---------------------------
def _to_chw_float01_from_bgr_np(img_bgr_np: np.ndarray, device=None) -> torch.Tensor:
    """
    np.uint8 BGR (H,W,3) -> torch.float32 RGB (C,H,W) di [0,1] pada device.
    Tanpa resize.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert img_bgr_np.ndim == 3 and img_bgr_np.shape[2] == 3, "butuh BGR (H,W,3)"
    # BGR->RGB
    img_rgb_np = cv2.cvtColor(img_bgr_np, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb_np).to(device=device, dtype=torch.float32) / 255.0   # (H,W,3) 0..1
    t = t.permute(2, 0, 1).contiguous()  # (C,H,W)
    return t

def _rgb_to_lab_torch(x_chw: torch.Tensor) -> torch.Tensor:
    """
    sRGB (C,H,W, [0..1]) -> CIELAB (C,H,W), D65. Semua di device yang sama.
    """
    if x_chw.shape[0] == 1:  # grayscale -> replikasi ke RGB netral
        x_chw = x_chw.repeat(3,1,1)
    R, G, B = x_chw[0], x_chw[1], x_chw[2]

    # inverse gamma sRGB -> linear
    a = (R <= 0.04045).to(R.dtype)
    r = a * (R / 12.92) + (1 - a) * (((R + 0.055) / 1.055).clamp(min=0) ** 2.4)
    a = (G <= 0.04045).to(G.dtype)
    g = a * (G / 12.92) + (1 - a) * (((G + 0.055) / 1.055).clamp(min=0) ** 2.4)
    a = (B <= 0.04045).to(B.dtype)
    b = a * (B / 12.92) + (1 - a) * (((B + 0.055) / 1.055).clamp(min=0) ** 2.4)

    # RGB->XYZ (D65)
    X = 0.4124564*r + 0.3575761*g + 0.1804375*b
    Y = 0.2126729*r + 0.7151522*g + 0.0721750*b
    Z = 0.0193339*r + 0.1191920*g + 0.9503041*b

    # normalize white point
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = X / Xn
    y = Y / Yn
    z = Z / Zn

    eps = 216/24389
    kappa = 24389/27

    def f(t):
        c = (t > eps).to(t.dtype)
        return c * (t ** (1/3)) + (1 - c) * ((kappa * t + 16) / 116)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116*fy - 16          # [0..100]
    a = 500*(fx - fy)        # ~[-128..128]
    b = 200*(fy - fz)        # ~[-128..128]
    return torch.stack([L, a, b], 0)

def _percentile_torch(x: torch.Tensor, q: float) -> torch.Tensor:
    """
    Percentile sederhana di GPU. x: 2D/3D -> skalar.
    """
    v = x.reshape(-1)
    n = v.numel()
    if n == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    k = int(round(q/100.0 * (n-1)))
    vals, _ = torch.sort(v)
    return vals[k]

def _alpha_trim_mean_var_torch(v: torch.Tensor, alpha: float = 0.1):
    """
    Mean & var alpha-trim di GPU.
    """
    v = v.reshape(-1)
    n = v.numel()
    if n == 0:
        z = torch.tensor(0.0, device=v.device, dtype=v.dtype)
        return z, z
    t = int(round(alpha * n / 2.0))
    v_sorted, _ = torch.sort(v)
    if 2*t >= n:
        t = max(0, n//2 - 1)
    v_trim = v_sorted[t:n-t] if t > 0 else v_sorted
    m = v_trim.mean()
    var = ((v_trim - m) ** 2).mean()
    return m, var

# ---------------------------
# UCIQE (CUDA)
# ---------------------------
def getUCIQE_cuda(img_BGR: np.ndarray, device=None) -> float:
    """
    UCIQE = 0.4680*sigma_c + 0.2745*con_l + 0.2576*avg_sat
    - Pakai LAB ter-normalisasi (Ln=L*/100, an=a*/128, bn=b*/128)
    - Operasi di GPU (kecuali konversi BGR->RGB yang trivial)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # BGR np -> RGB tensor (C,H,W) [0,1] di device
    x = _to_chw_float01_from_bgr_np(img_BGR, device=device)

    # ke Lab (C,H,W) di device
    Lab = _rgb_to_lab_torch(x)
    L_star, a_star, b_star = Lab[0], Lab[1], Lab[2]

    # normalisasi sesuai definisi UCIQE
    Ln = (L_star / 100.0).clamp(0, 1)
    an = (a_star / 128.0).clamp(-1, 1)
    bn = (b_star / 128.0).clamp(-1, 1)

    Cn = torch.sqrt(an*an + bn*bn)
    sigma_c = Cn.std(unbiased=False)

    L1 = _percentile_torch(Ln, 1.0)
    L99 = _percentile_torch(Ln, 99.0)
    con_l = L99 - L1

    s = Cn / (Ln + 1e-6)
    avg_sat = s.mean()

    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    uciqe = c1*sigma_c + c2*con_l + c3*avg_sat
    return float(uciqe.item())

# ---------------------------
# UIQM (CUDA)
# ---------------------------
def _eme_unfold(x2d: torch.Tensor, win: int = 10) -> torch.Tensor:
    """
    EME/log natural yang stabil via unfold (non-overlap).
    x2d: (H,W) float (disarankan skala 0..255)
    """
    H, W = x2d.shape
    ny = H // win
    nx = W // win
    if ny == 0 or nx == 0:
        return torch.tensor(0.0, device=x2d.device, dtype=x2d.dtype)

    Ht = ny * win
    Wt = nx * win
    x = x2d[:Ht, :Wt].unsqueeze(0).unsqueeze(0)  # (1,1,Ht,Wt)

    patches = x.unfold(2, win, win).unfold(3, win, win)  # (1,1,ny,nx,win,win)
    pmin = patches.amin(dim=(-1, -2))
    pmax = patches.amax(dim=(-1, -2))

    pmin = torch.clamp(pmin, min=1.0)
    pmax = torch.maximum(pmax, pmin + 1e-6)

    ln_ratio = torch.log(pmax / pmin)  # natural log
    w = 2.0 / (nx * ny)
    return (w * ln_ratio).mean()

def _uicm_cuda(x_rgb01: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """
    UICM di GPU: alpha-trim pada RG & YB (x: RGB (C,H,W) [0,1])
    """
    R, G, B = x_rgb01[0], x_rgb01[1], x_rgb01[2]
    RG = R - G
    YB = 0.5*(R + G) - B
    mu_rg, var_rg = _alpha_trim_mean_var_torch(RG, alpha)
    mu_yb, var_yb = _alpha_trim_mean_var_torch(YB, alpha)
    lam1, lam2 = -0.0268, 0.1586
    return lam1 * torch.sqrt(mu_rg*mu_rg + mu_yb*mu_yb + 1e-12) + \
           lam2 * torch.sqrt(var_rg + var_yb + 1e-12)

def _uism_cuda(x_rgb01: torch.Tensor, win: int = 10) -> torch.Tensor:
    """
    UISM di GPU: Sobel magnitude -> EME per channel -> kombinasikan (BT.601).
    """
    C, H, W = x_rgb01.shape
    # konversi ke 0..255 untuk EME
    x255 = (x_rgb01 * 255.0).clamp(0, 255)

    # Sobel kernel
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x_rgb01.device, dtype=torch.float32).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=x_rgb01.device, dtype=torch.float32).view(1,1,3,3)

    weights = [0.299, 0.587, 0.114] if C >= 3 else [1.0]
    total = 0.0
    for c in range(C):
        ch = x255[c:c+1].unsqueeze(0)  # (1,1,H,W)
        gx = F.conv2d(ch, kx, padding=1)
        gy = F.conv2d(ch, ky, padding=1)
        mag = torch.sqrt(gx*gx + gy*gy + 1e-12).squeeze(0).squeeze(0)  # (H,W)
        # normalisasi ulang ke 0..255 untuk EME
        mmax = mag.max()
        mag255 = (mag / (mmax + 1e-6)) * 255.0
        total = total + weights[min(c, len(weights)-1)] * _eme_unfold(mag255, win)
    return torch.as_tensor(total, device=x_rgb01.device, dtype=torch.float32)

def _uiconm_cuda(x_rgb01: torch.Tensor, win: int = 10) -> torch.Tensor:
    """
    UIConM di GPU: EME pada luminance Y (BT.601).
    """
    if x_rgb01.shape[0] >= 3:
        Y = 0.299*x_rgb01[0] + 0.587*x_rgb01[1] + 0.114*x_rgb01[2]
    else:
        Y = x_rgb01[0]
    Y255 = (Y * 255.0).clamp(0, 255)
    return _eme_unfold(Y255, win)

def getUIQM_cuda(img_BGR: np.ndarray, window_size: int = 10, device=None):
    """
    Hitung UICM, UISM, UIConM, UIQM di GPU.
    - img_BGR: np.uint8 (H,W,3), tanpa resize
    - window_size: ukuran blok EME (mis. 10)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # BGR np -> RGB tensor (C,H,W) [0,1]
    x_rgb01 = _to_chw_float01_from_bgr_np(img_BGR, device=device)

    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uicm = _uicm_cuda(x_rgb01, alpha=0.1)
    uism = _uism_cuda(x_rgb01, win=window_size)
    uiconm = _uiconm_cuda(x_rgb01, win=window_size)
    uiqm = c1*uicm + c2*uism + c3*uiconm
    return float(uicm.item()), float(uism.item()), float(uiconm.item()), float(uiqm.item())

def getScore_cuda(image_path, device=None):
    """
    Versi GPU dari getScore:
    - image_path: string atau Path ke file gambar (misalnya PNG/JPG)
    - device: PyTorch device (default: CUDA jika tersedia)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Baca gambar langsung dari path (OpenCV baca BGR)
    head, filename = os.path.split(image_path)

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Gagal membaca file: {image_path}")

    # Hitung metrik
    UICM, UISM, UIConM, UIQM = getUIQM_cuda(img_bgr, window_size=10, device=device)
    UCIQE = getUCIQE_cuda(img_bgr, device=device)

    return filename, UIQM, UCIQE

