import numpy as np  # type: ignore
from PIL import Image  # type: ignore
import cupy as cp  # type: ignore
import torch  # type: ignore



def compensate_R_cuda(image):
    """
    GPU (CUDA) version of compensate_R using CuPy.
    Input:  PIL.Image or numpy ndarray (H, W, 3) in RGB order
    Output: numpy ndarray (H, W, 3) in BGR order (uint8), same as original
    """
    # --- Load to host ndarray (float32) ---
    if isinstance(image, Image.Image):
        arr = np.array(image, dtype=np.float32)  # RGB
    elif isinstance(image, np.ndarray):
        if image.dtype != np.float32:
            arr = image.astype(np.float32, copy=False)
        else:
            arr = image
    else:
        raise TypeError("image must be PIL.Image or numpy.ndarray")

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")

    # --- Move to GPU ---
    dimg = cp.asarray(arr, dtype=cp.float32)  # RGB on device
    R = dimg[:, :, 0]
    G = dimg[:, :, 1]
    B = dimg[:, :, 2]

    # --- Per-channel min/max (from original, before normalization) ---
    minR, maxR = cp.min(R), cp.max(R)
    minG, maxG = cp.min(G), cp.max(G)
    minB, maxB = cp.min(B), cp.max(B)

    # --- Normalize to (0,1) with epsilon to avoid /0 ---
    eps = cp.float32(1e-6)
    Rn = (R - minR) / cp.maximum(maxR - minR, eps)
    Gn = (G - minG) / cp.maximum(maxG - minG, eps)
    Bn = (B - minB) / cp.maximum(maxB - minB, eps)

    # --- Means ---
    meanR = cp.mean(Rn)
    meanG = cp.mean(Gn)
    # meanB not used in original formula

    # --- Compensate Red channel (vectorized) ---
    Rn_new = Rn + (meanG - meanR) * (1.0 - Rn) * Gn

    # --- Scale back to original ranges and clamp ---
    R_scaled = cp.clip(Rn_new * maxR, 0, maxR)
    G_scaled = cp.clip(Gn * maxG, 0, maxG)
    B_scaled = cp.clip(Bn * maxB, 0, maxB)

    # --- Stack and return in BGR (to match your original output) ---
    out_gpu = cp.stack([B_scaled, G_scaled, R_scaled], axis=2).astype(cp.uint8)
    out = cp.asnumpy(out_gpu)  # back to host

    return out

def compensate_R_torch(image_path, device: str | None = None):
    """
    PyTorch (CUDA-capable) version of compensate_R.
    Input : PIL.Image or numpy ndarray (H, W, 3) in RGB
    Output: numpy ndarray (H, W, 3) in BGR (uint8), same as original
    """
    # --- Select device ---

    image = Image.open(image_path).convert('RGB')
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- To host ndarray (float32) ---
    if isinstance(image, Image.Image):
        arr = np.array(image, dtype=np.float32)  # RGB
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float32, copy=False)
    else:
        raise TypeError("image must be PIL.Image or numpy.ndarray")

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")

    # --- To torch tensor on device ---
    with torch.no_grad():
        t = torch.from_numpy(arr).to(device=device)           # H, W, 3 (RGB), float32
        R = t[..., 0]
        G = t[..., 1]
        B = t[..., 2]

        # Per-channel min/max (from original data)
        minR, maxR = torch.min(R), torch.max(R)
        minG, maxG = torch.min(G), torch.max(G)
        minB, maxB = torch.min(B), torch.max(B)

        eps = torch.tensor(1e-6, device=device, dtype=torch.float32)

        # Normalize to (0,1)
        Rn = (R - minR) / torch.maximum(maxR - minR, eps)
        Gn = (G - minG) / torch.maximum(maxG - minG, eps)
        Bn = (B - minB) / torch.maximum(maxB - minB, eps)

        # Means
        meanR = torch.mean(Rn)
        meanG = torch.mean(Gn)

        # Compensate Red channel
        Rn_new = Rn + (meanG - meanR) * (1.0 - Rn) * Gn

        # Scale back to original ranges and clamp
        R_scaled = torch.clamp(Rn_new * maxR, 0, maxR)
        G_scaled = torch.clamp(Gn * maxG, 0, maxG)
        B_scaled = torch.clamp(Bn * maxB, 0, maxB)

        # Stack back in BGR order (to match your original output)
        out = torch.stack([B_scaled, G_scaled, R_scaled], dim=-1).to(torch.uint8)

    return out.cpu().numpy()