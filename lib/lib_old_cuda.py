import numpy as np # type: ignore
from PIL import Image # type: ignore
import torch # type: ignore

def compensate_R_torch(image, device: str | None = None):
    """
    PyTorch (CUDA-capable) version of compensate_R.
    Input : PIL.Image or numpy ndarray (H, W, 3) in RGB
    Output: numpy ndarray (H, W, 3) in BGR (uint8), same as original
    """
    # # --- Select device ---
    # if device is None:
    #     device = "cuda" if torch.cuda.is_available() else "cpu"

    # # --- To host ndarray (float32) ---
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
        meanB = torch.mean(Bn)

        # Compensate Red channel
        Rn_new = Rn + (meanG - meanR) * (1.0 - Rn) * Gn
        Bn_new = Bn + (meanG - meanB) * (1.0 - Bn) * Gn

        # Scale back to original ranges and clamp
        R_scaled = torch.clamp(Rn_new * maxR, 0, maxR)
        G_scaled = torch.clamp(Gn * maxG, 0, maxG)
        B_scaled = torch.clamp(Bn_new * maxB, 0, maxB)

        # Stack back in BGR order (to match your original output)
        out = torch.stack([R_scaled, G_scaled, B_scaled], dim=-1).to(torch.uint8)

    return out.cpu().numpy()
