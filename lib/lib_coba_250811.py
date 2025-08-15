import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch

# --- Fungsi GPU ---
def compensate_R_torch(image, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(image, Image.Image):
        arr = np.array(image, dtype=np.float32)  # RGB
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float32, copy=False)
    else:
        raise TypeError("image must be PIL.Image or numpy.ndarray")

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("image must memiliki shape (H, W, 3)")

    with torch.no_grad():
        t = torch.from_numpy(arr).to(device=device)
        R, G, B = t[..., 0], t[..., 1], t[..., 2]

        minR, maxR = torch.min(R), torch.max(R)
        minG, maxG = torch.min(G), torch.max(G)
        minB, maxB = torch.min(B), torch.max(B)

        eps = torch.tensor(1e-6, device=device, dtype=torch.float32)

        Rn = (R - minR) / torch.maximum(maxR - minR, eps)
        Gn = (G - minG) / torch.maximum(maxG - minG, eps)
        Bn = (B - minB) / torch.maximum(maxB - minB, eps)

        meanR = torch.mean(Rn)
        meanG = torch.mean(Gn)

        Rn_new = Rn + (meanG - meanR) * (1.0 - Rn) * Gn

        R_scaled = torch.clamp(Rn_new * maxR, 0, maxR)
        G_scaled = torch.clamp(Gn * maxG, 0, maxG)
        B_scaled = torch.clamp(Bn * maxB, 0, maxB)

        out = torch.stack([R_scaled, G_scaled, B_scaled], dim=-1).to(torch.uint8)

    return out.cpu().numpy() 


# --- Fungsi proses folder dan simpan output ---
def process_folder(input_dir, output_dir, pattern="*.png", device=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True) # Membuat direktori output jika belum ada

    files = sorted(input_dir.glob(pattern))
    if not files:
        print("Tidak ada file ditemukan.")
        return

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device} | Jumlah file: {len(files)}")

    for idx, file_path in enumerate(files, 1):
        try:
            img = Image.open(file_path).convert("RGB")
            out_bgr = compensate_R_torch(img, device=device)

            # Ubah BGR -> RGB sebelum simpan
            out_rgb = out_bgr[:, :, ::-1]
            out_img = Image.fromarray(out_rgb)

            save_path = output_dir / file_path.name
            out_img.save(save_path, format="PNG")

            print(f"[{idx}/{len(files)}] ✅ {file_path.name} disimpan ke {save_path}")
        except Exception as e:
            print(f"[{idx}/{len(files)}] ❌ Gagal {file_path.name} -> {e}")
