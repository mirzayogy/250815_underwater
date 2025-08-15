import torch  # type: ignore
import torchvision.io as io  # type: ignore
import torchvision.transforms.functional as TF  # type: ignore
from pathlib import Path


@torch.no_grad()
def gray_world_single_cuda(img_chw_uint8: torch.Tensor, eps=1e-6, device=None) -> torch.Tensor:
    """
    img_chw_uint8 : Tensor uint8 shape (C,H,W), RGB atau RGBA.
    return         : Tensor uint8 shape (C,H,W) dengan ukuran asli, RGB/RGBA sesuai input.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert img_chw_uint8.ndim == 3, "Input harus (C,H,W)."
    C, H, W = img_chw_uint8.shape
    assert C in (3, 4), "Hanya mendukung RGB (3) atau RGBA (4)."

    # Pisahkan alpha jika ada
    alpha = None
    if C == 4:
        alpha = img_chw_uint8[3:4]  # (1,H,W)
        img_chw_uint8 = img_chw_uint8[:3]  # ambil RGB
        C = 3

    # ke float32 di device
    x = img_chw_uint8.to(device=device, dtype=torch.float32)  # (3,H,W)

    R, G, B = x[0], x[1], x[2]
    meanR = R.mean()
    meanG = G.mean()
    meanB = B.mean()

    target = (meanR + meanG + meanB) / 3.0

    sR = target / (meanR + eps)
    sG = target / (meanG + eps)
    sB = target / (meanB + eps)

    out = torch.stack([R * sR, G * sG, B * sB], dim=0).clamp_(0, 255).to(torch.uint8)  # (3,H,W)

    # kembalikan alpha jika ada
    if alpha is not None:
        out = torch.cat([out, alpha.to(out.device)], dim=0)  # (4,H,W)

    return out

@torch.no_grad()
def gray_world_luma_single_cuda(img_chw_uint8: torch.Tensor, eps=1e-6, device=None) -> torch.Tensor:
    """
    img_chw_uint8 : Tensor uint8 shape (C,H,W), RGB atau RGBA.
    return         : Tensor uint8 shape (C,H,W) dengan ukuran asli, RGB/RGBA sesuai input.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert img_chw_uint8.ndim == 3, "Input harus (C,H,W)."
    C, H, W = img_chw_uint8.shape
    assert C in (3, 4), "Hanya mendukung RGB (3) atau RGBA (4)."

    # Pisahkan alpha jika ada
    alpha = None
    if C == 4:
        alpha = img_chw_uint8[3:4]  # (1,H,W)
        img_chw_uint8 = img_chw_uint8[:3]  # ambil RGB
        C = 3

    # ke float32 di device
    x = img_chw_uint8.to(device=device, dtype=torch.float32)  # (3,H,W)

    R, G, B = x[0], x[1], x[2]
    meanR = R.mean()
    meanG = G.mean()
    meanB = B.mean()

    target = (0.299*R + 0.587*G + 0.114*B).mean()

    sR = target / (meanR + eps)
    sG = target / (meanG + eps)
    sB = target / (meanB + eps)

    out = torch.stack([R * sR, G * sG, B * sB], dim=0).clamp_(0, 255).to(torch.uint8)  # (3,H,W)

    # kembalikan alpha jika ada
    if alpha is not None:
        out = torch.cat([out, alpha.to(out.device)], dim=0)  # (4,H,W)

    return out

def process_folder_gray_world_keep_size(input_folder, output_folder, device=None):
    """
    Memproses semua PNG di input_folder, menyimpan hasil ke output_folder
    dengan ukuran asli masing-masing gambar (tanpa resize).
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(list(input_folder.glob("*.png")))
    if not files:
        print("Tidak ada file PNG di folder input.")
        return

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    for idx, f in enumerate(files, 1):
        try:
            # read_image -> (C,H,W) uint8
            img = io.read_image(str(f))  # uint8, (C,H,W)
            # Pastikan C in {3,4}
            if img.shape[0] not in (3, 4):
                # Jika grayscale, naikkan ke RGB
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                else:
                    raise ValueError(f"Jumlah kanal tidak didukung: {img.shape[0]}")

            out = gray_world_single_cuda(img, device=device)  # (C,H,W) uint8
            save_path = output_folder / f.name
            io.write_png(out.cpu(), str(save_path))
            print(f"[{idx}/{len(files)}] OK: {f.name}")
        except Exception as e:
            print(f"[{idx}/{len(files)}] Gagal: {f.name} -> {e}")

    print("Selesai memproses semua gambar.")

def gray_world_torch(x, eps=1e-6, return_uint8=True, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    if x.ndim == 3:
        x = x.unsqueeze(0)
    assert x.shape[-1] == 3, "Channel terakhir harus RGB (3)."

    if x.dtype == torch.uint8:
        x = x.to(device=device, dtype=torch.float32)
    else:
        x = x.to(device=device).float()

    R = x[..., 0]
    G = x[..., 1]
    B = x[..., 2]

    meanR = R.mean(dim=(1, 2), keepdim=True)
    meanG = G.mean(dim=(1, 2), keepdim=True)
    meanB = B.mean(dim=(1, 2), keepdim=True)

    target = (meanR + meanG + meanB) / 3.0

    sR = target / (meanR + eps)
    sG = target / (meanG + eps)
    sB = target / (meanB + eps)

    scale = torch.cat([sR, sG, sB], dim=-1)
    out = x * scale
    out = out.clamp_(0, 255)

    if return_uint8:
        out = out.to(torch.uint8)

    if out.shape[0] == 1:
        out = out.squeeze(0)

    return out


def process_folder_gray_world(
    input_folder, output_folder, batch_size=8, resize_size=(256, 256), device=None
):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(list(input_folder.glob("*.png")))
    if not files:
        print("Tidak ada file PNG di folder input.")
        return

    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        batch_images = []
        for f in batch_files:
            img = io.read_image(str(f)).permute(1, 2, 0)  # (H,W,3)
            img = TF.resize(img.permute(2, 0, 1), resize_size).permute(1, 2, 0)  # resize
            batch_images.append(img)

        batch_tensor = torch.stack(batch_images, dim=0)  # (N,H,W,3)
        result_batch = gray_world_torch(batch_tensor, device=device)

        for img_tensor, file_path in zip(result_batch, batch_files):
            save_path = output_folder / file_path.name
            io.write_png(img_tensor.permute(2, 0, 1).cpu(), str(save_path))

        print(f"Proses batch {i//batch_size+1} selesai ({len(batch_files)} gambar).")

    print("Semua gambar selesai diproses.")


def gray_world_torch_luma(x, use_luma_target=True, eps=1e-6, return_uint8=True, device=None):
    """
    x: torch.Tensor atau numpy array/PIL yang akan dikonversi ke tensor
       Bentuk: (H, W, 3) atau (N, H, W, 3), RGB
       Tipe: uint8 atau float
    use_luma_target: True -> target = mean luma; False -> mean RGB klasik
    return_uint8: True -> kembalikan uint8 [0,255]
    device: 'cuda'|'cpu'|None (None -> 'cuda' jika tersedia, else 'cpu')
    """
    # Pilih device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Pastikan tensor
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    # Pastikan bentuk jadi (N,H,W,3)
    if x.ndim == 3:
        x = x.unsqueeze(0)  # (1,H,W,3)
    assert x.shape[-1] == 3, "Channel terakhir harus 3 (RGB)."

    # Ke float32 di device target
    if x.dtype == torch.uint8:
        x = x.to(device=device, dtype=torch.float32)
    else:
        x = x.to(device=device).float()

    # Kanal terpilah (broadcast aman)
    R = x[..., 0]
    G = x[..., 1]
    B = x[..., 2]

    # Mean per gambar (N,1,1)
    meanR = R.mean(dim=(1,2), keepdim=True)
    meanG = G.mean(dim=(1,2), keepdim=True)
    meanB = B.mean(dim=(1,2), keepdim=True)

    if use_luma_target:
        luma = (0.299*R + 0.587*G + 0.114*B)
        target = luma.mean(dim=(1,2), keepdim=True)
    else:
        target = (meanR + meanG + meanB) / 3.0

    sR = target / (meanR + eps)
    sG = target / (meanG + eps)
    sB = target / (meanB + eps)

    # Gabung skala jadi (N,1,1,3) untuk broadcast
    scale = torch.cat([sR, sG, sB], dim=-1)  # (N,1,1,3)
    out = x * scale
    out = out.clamp_(0, 255)

    if return_uint8:
        out = out.to(torch.uint8)

    # Jika input awal 3D, keluarkan 3D lagi
    if out.shape[0] == 1:
        out = out.squeeze(0)

    return out  # RGB

def process_folder_gray_world_luma_keep_size(input_folder, output_folder, device=None):
    """
    Memproses semua PNG di input_folder, menyimpan hasil ke output_folder
    dengan ukuran asli masing-masing gambar (tanpa resize).
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(list(input_folder.glob("*.png")))
    if not files:
        print("Tidak ada file PNG di folder input.")
        return

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    for idx, f in enumerate(files, 1):
        try:
            # read_image -> (C,H,W) uint8
            img = io.read_image(str(f))  # uint8, (C,H,W)
            # Pastikan C in {3,4}
            if img.shape[0] not in (3, 4):
                # Jika grayscale, naikkan ke RGB
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                else:
                    raise ValueError(f"Jumlah kanal tidak didukung: {img.shape[0]}")

            out = gray_world_luma_single_cuda(img, device=device)  # (C,H,W) uint8
            save_path = output_folder / f.name
            io.write_png(out.cpu(), str(save_path))
            print(f"[{idx}/{len(files)}] OK: {f.name}")
        except Exception as e:
            print(f"[{idx}/{len(files)}] Gagal: {f.name} -> {e}")

    print("Selesai memproses semua gambar.")