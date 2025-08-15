import os, glob, csv
import cv2  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn.functional as F # type: ignore

# ==============================
# Util kecil
# ==============================
def _ensure_batched_img_bgr(img: torch.Tensor):
    """
    img: (..., H, W, 3) BGR; mendukung (H,W,3) atau (N,H,W,3).
    return: (N,H,W,3), added_batch(bool)
    """
    assert img.ndim in (3,4) and img.shape[-1] == 3, "Shape harus (...,H,W,3) BGR"
    added = False
    if img.ndim == 3:
        img = img.unsqueeze(0)
        added = True
    return img, added

def _to_gray_batched_bgr(img_bgr: torch.Tensor):
    """ BGR -> Gray, img_bgr: (N,H,W,3) """
    # OpenCV gray weights untuk BGR
    B = img_bgr[...,0]
    G = img_bgr[...,1]
    R = img_bgr[...,2]
    gray = 0.114*B + 0.587*G + 0.299*R
    return gray

# ==============================
# sRGB (BGR) -> CIE Lab (D65) murni di Torch
# ==============================
@torch.no_grad()
def bgr_to_lab_torch(img_bgr: torch.Tensor):
    """
    img_bgr: (H,W,3) atau (N,H,W,3), BGR 0..255 (float/uint8)
    return: L,a,b dengan L in [0,100], a,b ~ [-128,127] (skala CIE Lab 'riil', bukan versi OpenCV 0..255)
    """
    x, added = _ensure_batched_img_bgr(img_bgr)
    x = x.to(torch.float32) / 255.0

    # BGR -> RGB
    R = x[...,2]
    G = x[...,1]
    B = x[...,0]
    rgb = torch.stack([R,G,B], dim=-1)

    # sRGB -> linear RGB (inverse gamma)
    thr = 0.04045
    low = (rgb <= thr).to(rgb.dtype) * (rgb / 12.92)
    high = (rgb > thr).to(rgb.dtype) * torch.pow((rgb + 0.055)/1.055, 2.4)
    rgb_lin = low + high

    # RGB -> XYZ (D65)
    # matrix untuk sRGB D65
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=rgb_lin.dtype, device=rgb_lin.device)
    XYZ = torch.einsum('...c,dc->...d', rgb_lin, M)

    # Normalize by reference white D65
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    X = XYZ[...,0] / Xn
    Y = XYZ[...,1] / Yn
    Z = XYZ[...,2] / Zn

    # f(t) for Lab
    eps = 216/24389  # ~0.008856
    kappa = 24389/27 # ~903.3

    def f(t):
        t_cbrt = torch.pow(torch.clamp_min(t, 1e-12), 1/3)
        return torch.where(t > eps, t_cbrt, (kappa*t + 16.0)/116.0)

    fX, fY, fZ = f(X), f(Y), f(Z)

    L = 116.0*fY - 16.0               # 0..100
    a = 500.0*(fX - fY)               # ~-128..127
    b = 200.0*(fY - fZ)

    if added:
        L = L.squeeze(0); a = a.squeeze(0); b = b.squeeze(0)
    return L, a, b

# ==============================
# UCIQE
# ==============================
@torch.no_grad()
def uciqe_bgr_torch(img_bgr: torch.Tensor):
    """
    UCIQE = 0.4680*σ_L + 0.2745*c̄ + 0.2576*σ_h
    img_bgr: (H,W,3) atau (N,H,W,3) BGR 0..255
    return float (kalau N>1, rata-rata batch)
    """
    L, a, b = bgr_to_lab_torch(img_bgr)  # L in [0,100], a,b real Lab
    C = torch.sqrt(a*a + b*b)
    h = torch.atan2(b, a)  # [-pi, pi]

    # buat batch-aware
    if L.ndim == 2:
        L = L.unsqueeze(0); C = C.unsqueeze(0); h = h.unsqueeze(0)

    sigma_L = L.float().std(dim=(1,2), unbiased=False)
    c_bar   = C.float().mean(dim=(1,2))
    sigma_h = h.float().std(dim=(1,2), unbiased=False)

    uciqe = 0.4680*sigma_L + 0.2745*c_bar + 0.2576*sigma_h
    return float(uciqe.mean().item())

# ==============================
# UIQM (UICM, UISM, UIConM)
# ==============================
@torch.no_grad()
def _uicm_bgr_torch(img_bgr: torch.Tensor):
    """
    UICM approx via Ruderman rg/yb
    """
    x, added = _ensure_batched_img_bgr(img_bgr)
    B = x[...,0].float(); G = x[...,1].float(); R = x[...,2].float()
    rg = R - G
    yb = (R + G)/2.0 - B
    uicm = -(0.0268*torch.sqrt((rg*rg).mean(dim=(1,2))) +
             0.1586*torch.sqrt((yb*yb).mean(dim=(1,2))))
    return uicm.mean().item()

@torch.no_grad()
def _sobel_magnitude_mean(gray_batched: torch.Tensor):
    """
    gray_batched: (N,H,W) float
    return: mean magnitude per image (N,), lalu dirata-rata
    """
    N,H,W = gray_batched.shape
    g = gray_batched.view(N,1,H,W)

    # Sobel kernels
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=g.dtype, device=g.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=g.dtype, device=g.device).view(1,1,3,3)

    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy).squeeze(1)
    return mag.mean(dim=(1,2))

@torch.no_grad()
def _uism_bgr_torch(img_bgr: torch.Tensor):
    x, added = _ensure_batched_img_bgr(img_bgr)
    gray = _to_gray_batched_bgr(x).float()
    m = _sobel_magnitude_mean(gray)  # (N,)
    return m.mean().item()

@torch.no_grad()
def _uiconm_bgr_torch(img_bgr: torch.Tensor):
    x, added = _ensure_batched_img_bgr(img_bgr)
    gray = _to_gray_batched_bgr(x).float()
    std = gray.std(dim=(1,2), unbiased=False)
    return std.mean().item()

@torch.no_grad()
def uiqm_bgr_torch(img_bgr: torch.Tensor, c1=0.0282, c2=0.2953, c3=3.5753):
    """
    UIQM = c1*UICM + c2*UISM + c3*UIConM
    """
    uicm  = _uicm_bgr_torch(img_bgr)
    uism  = _uism_bgr_torch(img_bgr)
    uicon = _uiconm_bgr_torch(img_bgr)
    return float(c1*uicm + c2*uism + c3*uicon)

# ==============================
# Tenengrad & Entropy
# ==============================
@torch.no_grad()
def tenengrad_bgr_torch(img_bgr: torch.Tensor):
    x, added = _ensure_batched_img_bgr(img_bgr)
    gray = _to_gray_batched_bgr(x).float()
    N,H,W = gray.shape
    g = gray.view(N,1,H,W)

    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=g.dtype, device=g.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=g.dtype, device=g.device).view(1,1,3,3)

    gx = F.conv2d(g, kx, padding=1).squeeze(1)
    gy = F.conv2d(g, ky, padding=1).squeeze(1)
    ten = (gx*gx + gy*gy).mean(dim=(1,2))
    return float(ten.mean().item())

@torch.no_grad()
def entropy_gray_bgr_torch(img_bgr: torch.Tensor, bins: int = 256):
    """
    Entropi pada gray 0..255. Gunakan torch.histc untuk tiap gambar lalu rata-rata.
    """
    x, added = _ensure_batched_img_bgr(img_bgr)
    gray = torch.clamp(_to_gray_batched_bgr(x), 0, 255).float()

    ents = []
    for i in range(gray.shape[0]):
        g = gray[i]
        hist = torch.histc(g, bins=bins, min=0, max=255)
        p = hist / (hist.sum() + 1e-12)
        p = p[p > 0]
        ent = -(p * torch.log2(p)).sum()
        ents.append(ent)
    return float(torch.stack(ents).mean().item())


# ----------------------------------------------------
# Evaluasi satu folder → CSV + ringkasan mean ± std
# ----------------------------------------------------
@torch.no_grad()
def evaluate_folder_metrics_torch(
    input_dir: str,
    csv_out: str = "metrics.csv",
    device: str = "cuda",                      # "cuda" | "cpu" | "mps"
    patterns: tuple = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"),
    recursive: bool = False
):
    # pilih device
    if device == "cuda" and torch.cuda.is_available():
        dev = "cuda"
    elif device == "mps" and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"

    # kumpulkan file
    files = []
    for p in patterns:
        pattern = os.path.join(input_dir, "**", p) if recursive else os.path.join(input_dir, p)
        files.extend(glob.glob(pattern, recursive=recursive))
    files = sorted(files)

    if not files:
        print("[info] tidak ada file gambar di folder tersebut.")
        return

    print(f"[mulai] {len(files)} file, device={dev}")
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)

    rows = []
    for i, fp in enumerate(files, 1):
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] gagal baca: {fp}")
            continue

        t = torch.from_numpy(img.astype(np.float32)).to(dev)

        uciqe  = uciqe_bgr_torch(t)
        uiqm   = uiqm_bgr_torch(t)
        tng    = tenengrad_bgr_torch(t)
        ent    = entropy_gray_bgr_torch(t)

        rows.append([os.path.basename(fp), uciqe, uiqm, tng, ent])
        print(f"[{i}/{len(files)}] {os.path.basename(fp)}  UCIQE={uciqe:.4f}  UIQM={uiqm:.4f}  TNG={tng:.2f}  H={ent:.3f}")

    # tulis CSV
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file","UCIQE","UIQM","Tenengrad","Entropy"])
        w.writerows(rows)

    # ringkasan
    arr = np.array(rows, dtype=object)
    uc, uq, tg, en = arr[:,1].astype(float), arr[:,2].astype(float), arr[:,3].astype(float), arr[:,4].astype(float)

    def mean_std(x): 
        return float(np.mean(x)), float(np.std(x, ddof=0))

    m_uc, s_uc = mean_std(uc)
    m_uq, s_uq = mean_std(uq)
    m_tg, s_tg = mean_std(tg)
    m_en, s_en = mean_std(en)

    print("\n[RINGKASAN]")
    print(f"UCIQE     : {m_uc:.4f} ± {s_uc:.4f}")
    print(f"UIQM      : {m_uq:.4f} ± {s_uq:.4f}")
    print(f"Tenengrad : {m_tg:.2f} ± {s_tg:.2f}")
    print(f"Entropy   : {m_en:.3f} ± {s_en:.3f}")
    print(f"[saved] {csv_out}")


# ----------------------------------------------------
# (Opsional) Bandingkan dua folder (before vs after)
# Menjodohkan file berdasarkan NAMA yang sama.
# ----------------------------------------------------
@torch.no_grad()
def compare_two_folders_metrics_torch(
    before_dir: str,
    after_dir: str,
    csv_out: str = "metrics_compare.csv",
    device: str = "cuda",
    patterns: tuple = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
):
    # daftar 'before' dan cari pasangan di 'after'
    before_map = {}
    for p in patterns:
        for fp in glob.glob(os.path.join(before_dir, p)):
            before_map[os.path.basename(fp)] = fp

    after_map = {}
    for p in patterns:
        for fp in glob.glob(os.path.join(after_dir, p)):
            after_map[os.path.basename(fp)] = fp

    common = sorted(set(before_map.keys()) & set(after_map.keys()))
    if not common:
        print("[info] tidak ada pasangan file dengan nama yang sama.")
        return

    # device
    if device == "cuda" and torch.cuda.is_available():
        dev = "cuda"
    elif device == "mps" and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"

    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    rows = []
    print(f"[banding] {len(common)} pasangan, device={dev}")

    for i, fn in enumerate(common, 1):
        img_b = cv2.imread(before_map[fn], cv2.IMREAD_COLOR)
        img_a = cv2.imread(after_map[fn],  cv2.IMREAD_COLOR)
        if img_b is None or img_a is None:
            print(f"[skip] {fn} gagal baca")
            continue

        tb = torch.from_numpy(img_b.astype(np.float32)).to(dev)
        ta = torch.from_numpy(img_a.astype(np.float32)).to(dev)

        # sebelum
        b_uc, b_uq = uciqe_bgr_torch(tb), uiqm_bgr_torch(tb)
        b_tg, b_en = tenengrad_bgr_torch(tb), entropy_gray_bgr_torch(tb)

        # sesudah
        a_uc, a_uq = uciqe_bgr_torch(ta), uiqm_bgr_torch(ta)
        a_tg, a_en = tenengrad_bgr_torch(ta), entropy_gray_bgr_torch(ta)

        rows.append([fn, b_uc, a_uc, a_uc-b_uc,
                         b_uq, a_uq, a_uq-b_uq,
                         b_tg, a_tg, a_tg-b_tg,
                         b_en, a_en, a_en-b_en])

        print(f"[{i}/{len(common)}] {fn}  ΔUCIQE={a_uc-b_uc:+.4f}  ΔUIQM={a_uq-b_uq:+.4f}  ΔTNG={a_tg-b_tg:+.2f}  ΔH={a_en-b_en:+.3f}")

    # tulis CSV
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "file",
            "UCIQE_before","UCIQE_after","ΔUCIQE",
            "UIQM_before","UIQM_after","ΔUIQM",
            "Tenengrad_before","Tenengrad_after","ΔTenengrad",
            "Entropy_before","Entropy_after","ΔEntropy"
        ])
        w.writerows(rows)

    print(f"[saved] {csv_out}")
