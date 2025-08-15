#!/usr/bin/env python3
import csv
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch # type: ignore
import torch.nn.functional as F # type: ignore
from PIL import Image # type: ignore

# =======================
# Utils I/O
# =======================
def _load_png_uint8_chw(path: Path) -> torch.Tensor:
    """
    Baca PNG sebagai tensor uint8 (C,H,W) TANPA resize.
    Hanya mendukung mode 'RGB' atau 'L' (grayscale).
    """
    with Image.open(path) as im:
        im.load()
        if im.mode == "RGB":
            w, h = im.size
            t = torch.frombuffer(im.tobytes(), dtype=torch.uint8)
            t = t.view(h, w, 3).permute(2, 0, 1).contiguous()  # (3,H,W)
        elif im.mode == "L":
            w, h = im.size
            t = torch.frombuffer(im.tobytes(), dtype=torch.uint8)
            t = t.view(h, w).unsqueeze(0).contiguous()         # (1,H,W)
        else:
            raise ValueError(f"Mode tidak didukung: {im.mode}. Hanya RGB atau L.")
    return t

def _to_chw_float01(x: torch.Tensor) -> torch.Tensor:
    """Pastikan (C,H,W) float32 [0..1] tanpa mengubah ukuran."""
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.dtype == torch.uint8:
        x = x.to(torch.float32) / 255.0
    else:
        x = x.to(torch.float32)
        if x.max() > 1.5:
            x = x / 255.0
    return x

# =======================
# Warna & statistik
# =======================
def _rgb_to_lab(x_chw: torch.Tensor) -> torch.Tensor:
    """sRGB (C,H,W, [0..1]) -> CIELAB (C,H,W), iluminan D65."""
    if x_chw.shape[0] == 1:  # grayscale -> replikasi ke 3 kanal netral
        x_chw = x_chw.repeat(3,1,1)
    R, G, B = x_chw[0], x_chw[1], x_chw[2]

    def inv_gamma(u):
        a = (u <= 0.04045).to(u.dtype)
        return a * (u / 12.92) + (1 - a) * (((u + 0.055) / 1.055) ** 2.4)

    r = inv_gamma(R); g = inv_gamma(G); b = inv_gamma(B)
    # RGB->XYZ (D65)
    X = 0.4124564*r + 0.3575761*g + 0.1804375*b
    Y = 0.2126729*r + 0.7151522*g + 0.0721750*b
    Z = 0.0193339*r + 0.1191920*g + 0.9503041*b

    # Normalisasi titik putih D65
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = X / Xn; y = Y / Yn; z = Z / Zn

    eps = 216/24389
    kappa = 24389/27
    def f(t):
        c = (t > eps).to(t.dtype)
        return c * (t ** (1/3)) + (1 - c) * ((kappa * t + 16) / 116)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return torch.stack([L, a, b], 0)

def _percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    k = int(torch.clamp(torch.tensor(round(q/100.0 * (x.numel()-1))), 0, x.numel()-1))
    vals, _ = torch.sort(x.reshape(-1))
    return vals[k]

def _alpha_trim_mean_var(v: torch.Tensor, alpha: float = 0.1):
    v = v.reshape(-1)
    n = v.numel()
    if n == 0:
        return torch.tensor(0.0, device=v.device), torch.tensor(0.0, device=v.device)
    t = int(round(alpha * n / 2.0))
    if 2*t >= n:
        t = max(0, n//2 - 1)
    v_sorted, _ = torch.sort(v)
    v_trim = v_sorted[t:n-t] if t > 0 else v_sorted
    m = v_trim.mean()
    var = ((v_trim - m) ** 2).mean()
    return m, var

# =======================
# UCIQE
# =======================
def uciqe(img: torch.Tensor) -> torch.Tensor:
    """
    UCIQE = 0.4680*σ_c + 0.2745*con_l + 0.2576*μ_s
    img: (C,H,W) uint8/float; tanpa resize
    """
    x = _to_chw_float01(img)
    Lab = _rgb_to_lab(x)
    L, a, b = Lab[0], Lab[1], Lab[2]
    C = torch.sqrt(a*a + b*b)

    sigma_c = C.std(unbiased=False)
    L1 = _percentile(L, 1.0)
    L99 = _percentile(L, 99.0)
    con_l = L99 - L1
    s = C / (L + 1e-6)
    mu_s = s.mean()
    return 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s

# =======================
# UIQM (UICM, UISM, UIConM)
# =======================
def _eme(block: torch.Tensor, eps=1e-12) -> torch.Tensor:
    bmin = block.min()
    bmax = block.max()
    return 20.0 * torch.log10((bmax + eps) / (bmin + eps))

def _eme_map(img: torch.Tensor, k=8) -> torch.Tensor:
    H, W = img.shape[-2], img.shape[-1]
    by = max(1, H // k)
    bx = max(1, W // k)
    vals = []
    for y0 in range(0, H, by):
        for x0 in range(0, W, bx):
            y1 = min(H, y0 + by)
            x1 = min(W, x0 + bx)
            vals.append(_eme(img[..., y0:y1, x0:x1]))
    return torch.stack(vals).mean()

def _uicm(img: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    x = _to_chw_float01(img)
    if x.shape[0] >= 3:
        R, G, B = x[0], x[1], x[2]
    else:
        R = G = B = x[0]
    RG = R - G
    YB = 0.5 * (R + G) - B
    mu_rg, var_rg = _alpha_trim_mean_var(RG, alpha)
    mu_yb, var_yb = _alpha_trim_mean_var(YB, alpha)
    lam1, lam2 = -0.0268, 0.1586
    return lam1 * torch.sqrt(mu_rg*mu_rg + mu_yb*mu_yb + 1e-12) + \
           lam2 * torch.sqrt(var_rg + var_yb + 1e-12)

def _uism(img: torch.Tensor, k_blocks: int = 8) -> torch.Tensor:
    x = _to_chw_float01(img)
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    vals = []
    weights = [0.299, 0.587, 0.114] if x.shape[0] >= 3 else [1.0]
    for c in range(x.shape[0]):
        ch = x[c:c+1]
        gx = F.conv2d(ch.unsqueeze(0), sobel_x, padding=1).squeeze(0)
        gy = F.conv2d(ch.unsqueeze(0), sobel_y, padding=1).squeeze(0)
        mag = torch.sqrt(gx*gx + gy*gy + 1e-12)  # (1,H,W)
        eme_val = _eme_map(mag.squeeze(0), k=k_blocks)
        vals.append(weights[c] * eme_val)
    return torch.stack(vals).sum()

def _uiconm(img: torch.Tensor, k_blocks: int = 8) -> torch.Tensor:
    x = _to_chw_float01(img)
    if x.shape[0] >= 3:
        Y = 0.299*x[0] + 0.587*x[1] + 0.114*x[2]
    else:
        Y = x[0]
    return _eme_map(Y.unsqueeze(0), k=k_blocks)

def uiqm(img: torch.Tensor,
         c1: float = 0.0282, c2: float = 0.2953, c3: float = 3.5753,
         alpha: float = 0.1, k_blocks: int = 8) -> torch.Tensor:
    uicm = _uicm(img, alpha)
    uism = _uism(img, k_blocks)
    uiconm = _uiconm(img, k_blocks)
    return c1*uicm + c2*uism + c3*uiconm

# =======================
# Evaluasi folder -> CSV
# =======================
def evaluate_folder_uiq(
    in_dir: Path,
    out_csv: Path,
    recurse: bool = False,
    device: Optional[torch.device] = None,
) -> None:
    """
    Baca semua PNG dari in_dir (optionally rekursif), hitung UIQM & UCIQE per gambar,
    TANPA resize, dan simpan ke CSV.
    Kolom: filename, width, height, channels, uiqm, uciqe, error
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths: List[Path] = sorted(in_dir.rglob("*.png") if recurse else in_dir.glob("*.png"))
    if not paths:
        raise SystemExit("Tidak ada file PNG di folder tersebut.")

    rows: List[Dict] = []
    ok = 0

    for p in paths:
        try:
            img = _load_png_uint8_chw(p).to(device)   # (C,H,W) uint8
            C, H, W = img.shape[0], img.shape[1], img.shape[2]

            # Hitung metrik
            u_uiqm = uiqm(img)
            u_uciqe = uciqe(img)

            rows.append({
                "filename": str(p.relative_to(in_dir)),
                "width": W,
                "height": H,
                "channels": C,
                "uiqm": float(u_uiqm.item()),
                "uciqe": float(u_uciqe.item()),
                "error": ""
            })
            ok += 1
        except Exception as e:
            rows.append({
                "filename": str(p.relative_to(in_dir)),
                "width": "",
                "height": "",
                "channels": "",
                "uiqm": "",
                "uciqe": "",
                "error": str(e)
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename","width","height","channels","uiqm","uciqe","error"])
        w.writeheader()
        w.writerows(rows)

    print(f"Selesai. Berhasil: {ok}/{len(paths)}. Hasil: {out_csv}")

# =======================
# Contoh pakai (Jupyter/Script)
# =======================
if __name__ == "__main__":
    # Ganti path sesuai kebutuhan
    in_dir = Path("./input_pngs")
    out_csv = Path("./uiq_metrics.csv")
    evaluate_folder_uiq(in_dir, out_csv, recurse=False, device=None)
