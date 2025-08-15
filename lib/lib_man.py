from pathlib import Path
from PIL import Image # type: ignore
import numpy as np # type: ignore
import torch # type: ignore

# ---------------- Core: batch GPU ----------------
@torch.inference_mode()
def compensate_R_torch_batch(rgb_f32: torch.Tensor) -> torch.Tensor:
    """
    rgb_f32: torch.Tensor (N,H,W,3) float32, nilai 0..255, di device (cuda/cpu)
    return : torch.Tensor (N,H,W,3) uint8, RGB (langsung, bukan BGR)
    """
    assert rgb_f32.ndim == 4 and rgb_f32.shape[-1] == 3, "Expect (N,H,W,3)"
    # -> (N,3,H,W) agar gampang dioperasikan
    x = rgb_f32.permute(0, 3, 1, 2)  # (N,3,H,W)
    R, G, B = x[:, 0:1], x[:, 1:2], x[:, 2:2+1]  # (N,1,H,W)

    # min/max per-image, per-channel -> (N,1,1,1) agar broadcast
    def ch_stats(t):
        mn = t.flatten(2).min(dim=2).values.view(-1,1,1,1)
        mx = t.flatten(2).max(dim=2).values.view(-1,1,1,1)
        return mn, mx

    minR, maxR = ch_stats(R)
    minG, maxG = ch_stats(G)
    minB, maxB = ch_stats(B)

    eps = torch.tensor(1e-6, dtype=rgb_f32.dtype, device=rgb_f32.device)

    # normalize (0..1)
    Rn = (R - minR) / torch.maximum(maxR - minR, eps)
    Gn = (G - minG) / torch.maximum(maxG - minG, eps)
    Bn = (B - minB) / torch.maximum(maxB - minB, eps)

    # mean per-image (per-channel)
    meanR = Rn.mean(dim=(2,3), keepdim=True)  # (N,1,1,1)
    meanG = Gn.mean(dim=(2,3), keepdim=True)  # (N,1,1,1)

    # compensate red (adaptif per-image)
    Rn_new = Rn + (meanG - meanR) * (1.0 - Rn) * Gn

    # scale back & clamp ke rentang asli per-image, per-channel
    R_scaled = torch.clamp(Rn_new * maxR, 0, maxR)
    G_scaled = torch.clamp(Gn      * maxG, 0, maxG)
    B_scaled = torch.clamp(Bn      * maxB, 0, maxB)

    # susun kembali jadi RGB dan cast ke uint8
    y = torch.cat([R_scaled, G_scaled, B_scaled], dim=1)          # (N,3,H,W)
    y = y.permute(0, 2, 3, 1).contiguous().to(torch.uint8)        # (N,H,W,3)
    return y
# -------------------------------------------------

def _load_images_to_batch(file_paths, resize_to):
    """Load & resize ke numpy float32 (N,H,W,3) [0..255]."""
    arrs = []
    for p in file_paths:
        img = Image.open(p).convert("RGB")
        if resize_to is not None:
            img = img.resize(resize_to, Image.BICUBIC)
        arrs.append(np.asarray(img, dtype=np.float32))
    batch_np = np.stack(arrs, axis=0)  # (N,H,W,3)
    return batch_np

def process_folder_batch(
    input_dir,
    output_dir,
    resize_to=(512, 512),    # seragamkan ukuran
    batch_size=16,
    recursive=False,
    use_amp=False,           # mixed precision opsional
    device=None
):
    """
    Proses semua PNG di folder secara batch di GPU.
    - input_dir : folder sumber
    - output_dir: folder hasil
    - resize_to : (W,H) atau None (harus seragam untuk batch)
    - batch_size: ukuran batch
    - recursive : True pakai rglob
    - use_amp   : True untuk autocast (hemat VRAM, sering lebih cepat)
    - device    : 'cuda'/'cpu'/torch.device. Default auto.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp = Path(input_dir)
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)

    files = sorted(inp.rglob("*.png") if recursive else inp.glob("*.png"))
    if not files:
        print("Tidak ada PNG di folder input.")
        return 0

    # batching
    n = len(files)
    for start in range(0, n, batch_size):
        chunk_paths = files[start:start+batch_size]
        batch_np = _load_images_to_batch(chunk_paths, resize_to)      # (N,H,W,3) float32
        batch = torch.from_numpy(batch_np).to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out_rgb = compensate_R_torch_batch(batch)              # (N,H,W,3) uint8
        else:
            out_rgb = compensate_R_torch_batch(batch)

        out_cpu = out_rgb.cpu().numpy()                                # ke host untuk simpan
        for i, p in enumerate(chunk_paths):
            Image.fromarray(out_cpu[i]).save(outp / p.name, format="PNG")

        # opsional: kosongkan cache VRAM antar batch besar
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return n

# ===== Contoh pakai =====
# total = process_folder_batch(
#     input_dir="folder_input",
#     output_dir="folder_output",
#     resize_to=(512, 512),   # set ke None kalau kamu yakin semua gambar sudah seragam
#     batch_size=32,
#     recursive=False,
#     use_amp=True            # aktifkan kalau GPU mendukung FP16
# )
# print(f"Selesai: {total} file.")
