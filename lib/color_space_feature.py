import torch # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore

# ---------- Util: RGB -> HSV (torch, vektorized) ----------
def rgb_to_hsv_torch(rgb):
    """
    rgb: float tensor in [0,1], shape (..., 3, H, W)
    return hsv in [H in 0..1, S in 0..1, V in 0..1]
    """
    r, g, b = rgb[..., 0, :, :], rgb[..., 1, :, :], rgb[..., 2, :, :]

    maxc, _ = torch.max(rgb, dim=-3)
    minc, _ = torch.min(rgb, dim=-3)
    v = maxc
    deltac = maxc - minc + 1e-12

    # Saturation
    s = deltac / (maxc + 1e-12)

    # Hue
    # init
    h = torch.zeros_like(maxc)
    mask = deltac > 0

    # Cases
    rc = (((maxc - r) / deltac) * mask).masked_fill(~mask, 0)
    gc = (((maxc - g) / deltac) * mask).masked_fill(~mask, 0)
    bc = (((maxc - b) / deltac) * mask).masked_fill(~mask, 0)

    # r is max
    cond = (maxc == r) & mask
    h = torch.where(cond, (bc - gc) % 6.0, h)
    # g is max
    cond = (maxc == g) & mask
    h = torch.where(cond, (2.0 + rc - bc), h)
    # b is max
    cond = (maxc == b) & mask
    h = torch.where(cond, (4.0 + gc - rc), h)

    h = (h / 6.0) % 1.0  # normalize to [0,1)
    hsv = torch.stack([h, s, v], dim=-3)
    return hsv

# ---------- Util: RGB -> LAB via sRGB -> XYZ (D65) ----------
def _srgb_to_linear(c):
    # c in [0,1]
    a = 0.055
    return torch.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)

def rgb_to_xyz_torch(rgb):
    """
    rgb: float tensor in [0,1], shape (..., 3, H, W), sRGB, D65
    returns XYZ in range ~[0,1]
    """
    r, g, b = rgb[..., 0, :, :], rgb[..., 1, :, :], rgb[..., 2, :, :]
    r_lin = _srgb_to_linear(r)
    g_lin = _srgb_to_linear(g)
    b_lin = _srgb_to_linear(b)

    # sRGB D65 conversion matrix
    # [X]   [0.4124564 0.3575761 0.1804375][R]
    # [Y] = [0.2126729 0.7151522 0.0721750][G]
    # [Z]   [0.0193339 0.1191920 0.9503041][B]
    X = 0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin
    Y = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin
    Z = 0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin
    return torch.stack([X, Y, Z], dim=-3)

def xyz_to_lab_torch(xyz):
    """
    xyz: tensor (..., 3, H, W), D65 white
    returns LAB in the CIE Lab space (L in 0..100 approximately, a/b ~ -128..127)
    """
    # D65 white point
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    X = xyz[..., 0, :, :] / Xn
    Y = xyz[..., 1, :, :] / Yn
    Z = xyz[..., 2, :, :] / Zn

    eps = 216/24389  # ~0.008856
    kappa = 24389/27  # ~903.3

    def f(t):
        return torch.where(t > eps, t.pow(1/3), (kappa * t + 16) / 116)

    fx, fy, fz = f(X), f(Y), f(Z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.stack([L, a, b], dim=-3)

def rgb_to_lab_torch(rgb):
    return xyz_to_lab_torch(rgb_to_xyz_torch(rgb))

# ---------- Loader ----------
def _to_rgb_tensor(image):
    """
    Accepts: path (str/Path), PIL.Image, or numpy array (H,W,3 or H,W,4 or 3D/4D)
    Returns: float tensor [1, 3, H, W] in [0,1]
    """
    if isinstance(image, (str, bytes)):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:  # grayscale
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        image = Image.fromarray(arr.astype(np.uint8))
    else:
        raise TypeError("Input must be a file path, PIL.Image, or numpy array.")

    t = torch.from_numpy(np.array(image)).float() / 255.0     # [H, W, 3], 0..1
    t = t.permute(2, 0, 1).unsqueeze(0)                       # [1, 3, H, W]
    return t

# ---------- Main extractor ----------
def extract_color_stats_cuda(image):
    """
    Compute (on CUDA):
      - mean & var: R,G,B
      - mean & var: H,S,V
      - mean & var: L,a,b
      - mean & var: ratios R/G, G/B, B/R

    Returns dict of floats.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia. Pastikan PyTorch terinstall dengan dukungan CUDA.")

    device = torch.device('cuda')
    x = _to_rgb_tensor(image).to(device)  # [1,3,H,W], 0..1
    # RGB stats
    R, G, B = x[:, 0], x[:, 1], x[:, 2]
    rgb_mean = {
        'R': R.mean().item(),
        'G': G.mean().item(),
        'B': B.mean().item()
    }
    rgb_var = {
        'R': R.var(unbiased=False).item(),
        'G': G.var(unbiased=False).item(),
        'B': B.var(unbiased=False).item()
    }

    # HSV stats
    hsv = rgb_to_hsv_torch(x)  # [1,3,H,W]
    Hc, Sc, Vc = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    hsv_mean = {'H': Hc.mean().item(), 'S': Sc.mean().item(), 'V': Vc.mean().item()}
    hsv_var  = {'H': Hc.var(unbiased=False).item(), 'S': Sc.var(unbiased=False).item(), 'V': Vc.var(unbiased=False).item()}

    # LAB stats
    lab = rgb_to_lab_torch(x)  # [1,3,H,W]
    Lc, ac, bc = lab[:, 0], lab[:, 1], lab[:, 2]
    lab_mean = {'L': Lc.mean().item(), 'a': ac.mean().item(), 'b': bc.mean().item()}
    lab_var  = {'L': Lc.var(unbiased=False).item(), 'a': ac.var(unbiased=False).item(), 'b': bc.var(unbiased=False).item()}

    # Ratios
    eps = 1e-12
    rg = R / (G + eps)
    gb = G / (B + eps)
    br = B / (R + eps)
    ratio_mean = {
        'R_over_G': rg.mean().item(),
        'G_over_B': gb.mean().item(),
        'B_over_R': br.mean().item(),
    }
    ratio_var = {
        'R_over_G': rg.var(unbiased=False).item(),
        'G_over_B': gb.var(unbiased=False).item(),
        'B_over_R': br.var(unbiased=False).item(),
    }

    return {
        'rgb_mean': rgb_mean,
        'rgb_var': rgb_var,
        'hsv_mean': hsv_mean,
        'hsv_var': hsv_var,
        'lab_mean': lab_mean,
        'lab_var': lab_var,
        'ratio_mean': ratio_mean,
        'ratio_var': ratio_var,
    }

# ---------- Contoh pemakaian di Jupyter ----------
# stats = extract_color_stats_cuda("contoh_gambar.jpg")
# stats
