import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage  # tetap ada, tapi tidak dipakai lagi di versi Torch
import os
# =========================
# Tetap: PSNR/MSE (biarkan CPU supaya angkanya identik)
# =========================
def psnrmse(reference, original):
    R2 = np.amax(reference)**2
    MSE = np.sum(np.power(np.subtract(reference, original), 2))
    MSE /= (reference.size[0] * original.size[1])
    PSNR = 10*np.log10(R2/MSE)
    return PSNR, MSE

def get_psnr_mse(image_path, folder_name):
    image_reference_path = image_path.replace(folder_name, "reference-890")
    im_reference = Image.open(image_reference_path)
    im_raw = Image.open(image_path)
    psnr, mse = psnrmse(im_reference, im_raw)
    return psnr, mse

# =========================
# Tetap: getUCIQE (gunakan OpenCV LAB agar identik)
# =========================
def getUCIQE(img_BGR):
    img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
    coe_Metric = [0.4680, 0.2745, 0.2576]

    img_lum = img_LAB[:,:,0]/255.0
    img_a   = img_LAB[:,:,1]/255.0
    img_b   = img_LAB[:,:,2]/255.0

    # item-1
    chroma  = np.sqrt(np.square(img_a)+np.square(img_b))
    sigma_c = np.std(chroma)

    # item-2
    img_lum_flat = img_lum.flatten()
    sorted_index = np.argsort(img_lum_flat)
    top_index    = sorted_index[int(len(img_lum_flat)*0.99)]
    bottom_index = sorted_index[int(len(img_lum_flat)*0.01)]
    con_lum      = img_lum_flat[top_index] - img_lum_flat[bottom_index]

    # item-3
    chroma_flat = chroma.flatten()
    sat = np.divide(chroma_flat, img_lum_flat, out=np.zeros_like(chroma_flat, dtype=np.float64), where=img_lum_flat!=0)
    avg_sat = np.mean(sat)

    uciqe = sigma_c*coe_Metric[0] + con_lum*coe_Metric[1] + avg_sat*coe_Metric[2]
    return uciqe

# =========================
# Versi GPU (Torch) untuk UIQM & komponen-komponennya
# =========================
def _get_device(device=None):
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _torch_from_bgr_numpy(img_bgr_np, device):
    """
    img_bgr_np: numpy array BGR HxWx3 (0..255), dtype uint8/float
    return: torch float32 [H,W,3] 0..255 (sesuai x.astype(np.float32) di getUIQM asli)
    """
    if img_bgr_np.ndim != 3 or img_bgr_np.shape[2] != 3:
        raise ValueError("Gambar harus BGR 3-channel.")
    t = torch.from_numpy(img_bgr_np).to(device=device)
    if t.dtype != torch.float32:
        t = t.to(torch.float32)
    return t

# ---- mu_a dan s_a (Torch) ----
def mu_a_torch(x1d, alpha_L=0.1, alpha_R=0.1):
    # x1d: 1D tensor float32
    x_sorted, _ = torch.sort(x1d)
    K = x_sorted.numel()
    if K == 0:
        return x_sorted.new_tensor(0.0, dtype=x_sorted.dtype)
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    s = T_a_L
    e = K - T_a_R
    if e <= s:
        return x_sorted.mean()
    return x_sorted[s:e].mean()

def s_a_torch(x1d, mu):
    if x1d.numel() == 0:
        return x1d.new_tensor(0.0, dtype=x1d.dtype)
    return ((x1d - mu)**2).mean()

# ---- Sobel (Torch) meniru ndimage.sobel: kernel klasik + padding reflect) ----
def sobel_torch(x2d):
    """
    x2d: [H,W] float32 (0..255)
    return: magnitude [H,W] float32, diskalakan * 255.0 / max seperti versi asli
    """
    H, W = x2d.shape
    # padding reflect biar sesuai ndimage.sobel (mode='reflect')
    x = F.pad(x2d[None,None,...], (1,1,1,1), mode='reflect')  # [1,1,H+2,W+2]

    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],
                       [ 0, 0, 0],
                       [ 1, 2, 1]], dtype=x.dtype, device=x.device).view(1,1,3,3)

    dx = F.conv2d(x, kx)  # [1,1,H,W]
    dy = F.conv2d(x, ky)
    mag = torch.hypot(dx, dy).squeeze(0).squeeze(0)  # [H,W]
    maxv = mag.max()
    if maxv > 0:
        mag = mag * (255.0 / maxv)
    else:
        mag = mag * 0.0
    return mag

# ---- EME (Torch) ----
def eme_torch(x2d, window_size):
    """
    x2d: [H,W] float32
    """
    H, W = x2d.shape
    k1 = W // window_size
    k2 = H // window_size
    if k1 == 0 or k2 == 0:
        return x2d.new_tensor(0.0, dtype=x2d.dtype)

    x = x2d[:k2*window_size, :k1*window_size]
    xx = x[None, None, ...]  # [1,1,H,W]
    patches = F.unfold(xx, kernel_size=window_size, stride=window_size)  # [1,ws*ws, k1*k2]
    p_max, _ = patches.max(dim=1)  # [1, k1*k2]
    p_min, _ = patches.min(dim=1)

    eps = 1e-12
    # sesuai log(max/min), dengan perlindungan 0
    valid = (p_max > 0) & (p_min > 0)
    ratio = torch.zeros_like(p_max)
    ratio[valid] = p_max[valid] / (p_min[valid] + eps)
    val = torch.log(torch.clamp(ratio, min=eps)).sum()

    w = 2.0 / (k1 * k2)
    return (w * val).to(x2d.dtype)

# ---- UICM, UISM, UICONM (Torch) ----
def _uicm_torch(x_bgr):
    """
    x_bgr: [H,W,3] float32 (0..255)
    """
    R = x_bgr[...,2].reshape(-1)
    G = x_bgr[...,1].reshape(-1)
    B = x_bgr[...,0].reshape(-1)

    RG = R - G
    YB = (R + G)/2.0 - B

    mu_a_RG = mu_a_torch(RG)
    mu_a_YB = mu_a_torch(YB)
    s_a_RG  = s_a_torch(RG, mu_a_RG)
    s_a_YB  = s_a_torch(YB, mu_a_YB)

    l = torch.sqrt(mu_a_RG**2 + mu_a_YB**2)
    r = torch.sqrt(s_a_RG + s_a_YB)
    return (-0.0268*l) + (0.1586*r)

def _uism_torch(x_bgr):
    """
    x_bgr: [H,W,3] float32
    """
    R = x_bgr[...,2]
    G = x_bgr[...,1]
    B = x_bgr[...,0]

    Rs = sobel_torch(R)
    Gs = sobel_torch(G)
    Bs = sobel_torch(B)

    R_edge_map = Rs * R
    G_edge_map = Gs * G
    B_edge_map = Bs * B

    r_eme = eme_torch(R_edge_map, 10)
    g_eme = eme_torch(G_edge_map, 10)
    b_eme = eme_torch(B_edge_map, 10)

    lambda_r, lambda_g, lambda_b = 0.299, 0.587, 0.144
    return lambda_r*r_eme + lambda_g*g_eme + lambda_b*b_eme

def _uiconm_torch(x_bgr, window_size):
    """
    x_bgr: [H,W,3] float32
    Meniru loop blok: top=max-min (across all channels), bot=max+min, akumulasi alpha*(top/bot)^alpha*log(top/bot)
    """
    H, W = x_bgr.shape[:2]
    k1 = W // window_size
    k2 = H // window_size
    if k1 == 0 or k2 == 0:
        return x_bgr.new_tensor(0.0, dtype=x_bgr.dtype)

    x = x_bgr[:k2*window_size, :k1*window_size, :]            # [H',W',3]
    xx = x.permute(2,0,1)[None, ...]                          # [1,3,H',W']
    patches = F.unfold(xx, kernel_size=window_size, stride=window_size)  # [1, 3*ws*ws, k1*k2]

    ws2 = window_size*window_size
    blocks = patches.squeeze(0).T.reshape(-1, 3, ws2)  # [num_blocks, 3, ws2]
    max_c = blocks.max(dim=2).values  # [num_blocks, 3]
    min_c = blocks.min(dim=2).values  # [num_blocks, 3]

    max_ = max_c.max(dim=1).values    # [num_blocks]
    min_ = min_c.min(dim=1).values

    top = max_ - min_
    bot = max_ + min_

    eps = 1e-12
    valid = (bot != 0.0) & (top != 0.0) & (~torch.isnan(top)) & (~torch.isnan(bot))
    ratio = torch.zeros_like(top)
    ratio[valid] = top[valid] / (bot[valid] + eps)

    alpha = 1.0
    contrib = torch.zeros_like(top)
    contrib[valid] = (ratio[valid]**alpha) * torch.log(torch.clamp(ratio[valid], min=eps))
    val = contrib.sum()

    w = -1.0 / (k1 * k2)
    return (w * val).to(x_bgr.dtype)

def getUIQM_torch(x_bgr):
    """
    x_bgr: [H,W,3] float32 (0..255)
    """
    # c1..c3 sama seperti kode asli
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uicm   = _uicm_torch(x_bgr)
    uism   = _uism_torch(x_bgr)
    uiconm = _uiconm_torch(x_bgr, 10)
    uiqm   = c1*uicm + c2*uism + c3*uiconm
    return uicm, uism, uiconm, uiqm

# =========================
# API utama: getScore (GPU) – menjaga angka sedekat mungkin
# =========================
def getScore_cuda(pil_image, device=None):
    """
    Input: PIL image (RGB)
    Output: UICM, UISM, UIConM, UIQM, UCIQE  (float Python)
    - UIQM dihitung di Torch (GPU)
    - UCIQE tetap pakai OpenCV LAB (CPU) agar identik dengan versi asli
    """
    device = _get_device(device)

    # Sama seperti versi asli: cv2.cvtColor dari RGB->BGR (NumPy), lalu kirim ke Torch
    img_bgr_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # Untuk UIQM, versi asli melakukan x.astype(np.float32)
    x_bgr_t = _torch_from_bgr_numpy(img_bgr_np.astype(np.float32), device)

    UICM_t, UISM_t, UIConM_t, UIQM_t = getUIQM_torch(x_bgr_t)
    # UCIQE via OpenCV agar konsisten
    UCIQE = float(getUCIQE(img_bgr_np))

    return float(UIQM_t.item()), UCIQE

# =========================
# Wrapper kompatibel seperti sebelumnya
# =========================
def getScore(pil_image):
    # agar kompatibel dengan signature lama Anda
    return getScore_cuda(pil_image)

def getScore_from_path_cuda(image_path, device=None):
    pil = Image.open(image_path).convert("RGB")
    UIQM, UCIQE = getScore(pil)
    filename = os.path.basename(image_path)
    return filename, UIQM, UCIQE

