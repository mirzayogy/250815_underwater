from pathlib import Path
from PIL import Image # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
import torch.nn.functional as F # type: ignore


# hasilnya jelek
@torch.inference_mode()
def batch_resize_png_pytorch(
    input_folder,
    result_folder,
    resize_to=(512, 512),   # (W, H)
    batch_size=32,
    recursive=False,
    use_amp=True,           # mixed precision (hemat VRAM & biasanya lebih cepat)
    device=None
):
    """
    Resize seluruh PNG dari input_folder ke size 'resize_to' (W,H) secara batch di GPU,
    dan simpan hasil ke result_folder dengan nama file yang sama.

    Parameters
    ----------
    input_folder : str | Path
    result_folder: str | Path
    resize_to    : tuple[int,int]  -> (width, height)
    batch_size   : int
    recursive    : bool            -> True untuk memproses subfolder juga
    use_amp      : bool            -> True aktifkan autocast (CUDA fp16)
    device       : torch.device | str | None

    Return
    ------
    int : jumlah file yang berhasil diproses
    """
    device = torch.device(device) if isinstance(device, str) else (device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    in_dir  = Path(input_folder)
    out_dir = Path(result_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.rglob("*.png") if recursive else in_dir.glob("*.png"))
    if not files:
        print("Tidak ada file PNG di folder input.")
        return 0

    Wt, Ht = int(resize_to[0]), int(resize_to[1])
    total = 0

    # Helper: pad ke (Hmax, Wmax) kanan & bawah
    def pad_to(t, Hmax, Wmax):
        # t: (3,H,W)
        _, H, W = t.shape
        pad_w = Wmax - W
        pad_h = Hmax - H
        if pad_w == 0 and pad_h == 0:
            return t
        return F.pad(t, (0, pad_w, 0, pad_h), mode="replicate")

    # Proses per-batch
    for s in range(0, len(files), batch_size):
        chunk = files[s:s+batch_size]

        # 1) Load ke CPU (ukuran bervariasi)
        imgs_cpu = []
        sizes = []
        for p in chunk:
            img = Image.open(p).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0    # (H,W,3) [0,1]
            ten = torch.from_numpy(arr).permute(2,0,1)         # (3,H,W)
            imgs_cpu.append(ten)
            sizes.append(ten.shape[1:])                        # (H,W)

        # 2) Tentukan Hmax,Wmax di batch, pad & pindah ke device, lalu stack
        Hmax = max(h for h, _ in sizes)
        Wmax = max(w for _, w in sizes)

        tensors = []
        for ten in imgs_cpu:
            tensors.append(pad_to(ten, Hmax, Wmax))
        batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)  # (N,3,Hmax,Wmax)

        # 3) Resize di GPU (bicubic)
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = F.interpolate(batch, size=(Ht, Wt), mode="bicubic", align_corners=False)
        else:
            out = F.interpolate(batch, size=(Ht, Wt), mode="bicubic", align_corners=False)

        # 4) Simpan kembali ke PNG (RGB uint8)
        out = (out.clamp(0,1) * 255.0).to(torch.uint8).permute(0,2,3,1).cpu().numpy()  # (N,Ht,Wt,3)
        for i, p in enumerate(chunk):
            Image.fromarray(out[i]).save(out_dir / p.name, format="PNG")

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        total += len(chunk)

    print(f"Selesai: {total} file -> {out_dir.resolve()}")
    return total

# ===== Contoh pakai =====
# batch_resize_png_pytorch(
#     input_folder="input_png",
#     result_folder="output_png",
#     resize_to=(512, 512),
#     batch_size=32,
#     recursive=False,
#     use_amp=True
# )


# Helper untuk resize folder PNG pakai PyTorch GPU
@torch.inference_mode()
def resize_png_folder_pytorch(input_folder, result_folder, resize_to=(512, 512), device=None):
    """
    Resize semua file PNG di input_folder ke ukuran resize_to menggunakan PyTorch GPU
    dan simpan hasilnya ke result_folder.

    Parameters:
    - input_folder (str/Path) : Folder sumber PNG
    - result_folder (str/Path): Folder hasil resize
    - resize_to (tuple)       : (width, height) hasil resize
    - device (torch.device)   : GPU/CPU (default: auto pilih GPU kalau ada)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_path = Path(input_folder)
    result_path = Path(result_folder)
    result_path.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob("*.png"))
    if not files:
        print("❌ Tidak ada file PNG di folder input.")
        return 0

    for idx, f in enumerate(files, 1):
        # Baca gambar dan konversi ke tensor
        img = Image.open(f).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0  # Normalisasi [0,1]
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)

        # Resize pakai interpolate GPU
        resized = torch.nn.functional.interpolate(
            tensor, size=(resize_to[1], resize_to[0]), mode="bicubic", align_corners=False
        )

        # Kembali ke numpy uint8 RGB
        resized_np = (resized.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Simpan hasil
        Image.fromarray(resized_np).save(result_path / f.name, format="PNG")
        # print(f"[{idx}/{len(files)}] ✅ {f.name} resized -> {resize_to}")

    print(f"🎯 Selesai: {len(files)} file disimpan di '{result_path}'")
    return len(files)

# ===== Contoh pakai =====
# resize_png_folder_pytorch("input_png", "output_png", resize_to=(512, 512))
