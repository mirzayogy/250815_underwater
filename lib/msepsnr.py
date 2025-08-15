#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import torch # type: ignore
from PIL import Image # type: ignore

# ---------- Util tensor ----------
def _ensure_tensor_chw_uint8(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL image to torch.Tensor shape (C,H,W) with dtype uint8.
    Tidak mengubah ukuran; hanya ubah representasi & channel order.
    """
    # Biarkan mode asli jika 1 channel (L) atau 3 channel (RGB)
    if img.mode == "RGB":
        arr = torch.frombuffer(img.tobytes(), dtype=torch.uint8)
        w, h = img.size
        arr = arr.view(h, w, 3).permute(2, 0, 1).contiguous()  # (C,H,W)
    elif img.mode == "L":
        arr = torch.frombuffer(img.tobytes(), dtype=torch.uint8)
        w, h = img.size
        arr = arr.view(h, w).unsqueeze(0).contiguous()  # (1,H,W)
    else:
        # Jangan paksa konversi ke RGB agar jumlah kanal tetap terjaga antar pasangan
        raise ValueError(f"Mode gambar tidak didukung: {img.mode}. Gunakan RGB atau L.")
    return arr

def _ensure_tensor4d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:  # (H,W)
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:  # (C,H,W)
        x = x.unsqueeze(0)
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(f"Bentuk tensor tidak didukung: {x.shape}")
    return x

def _infer_data_range(x: torch.Tensor) -> float:
    # x diharapkan uint8 atau float
    if x.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        return 255.0
    xmax = x.detach().max().item()
    return 255.0 if xmax > 1.5 else 1.0

def mse_per_image(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x4 = _ensure_tensor4d(x)
    y4 = _ensure_tensor4d(y)
    if x4.shape != y4.shape:
        raise ValueError(f"Ukuran/shape harus sama: {x4.shape} vs {y4.shape}")
    x4 = x4.to(torch.float32)
    y4 = y4.to(torch.float32)
    return ((x4 - y4) ** 2).mean(dim=(1, 2, 3))

def psnr_from_mse(mse_vals: torch.Tensor, data_range: float) -> torch.Tensor:
    eps = 1e-12
    return 10.0 * torch.log10((data_range ** 2) / torch.clamp(mse_vals, min=eps))

# ---------- I/O & pairing ----------
def list_images(folder: Path, exts: Tuple[str, ...]) -> Dict[str, Path]:
    d = {}
    for ext in exts:
        for p in folder.rglob(f"*{ext}"):
            d[p.name] = p
    return d

def load_image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        im.load()
        return _ensure_tensor_chw_uint8(im)

# ---------- Main evaluation ----------
def evaluate_folders(
    ref_dir: Path,
    pred_dir: Path,
    out_csv: Path,
    exts: Tuple[str, ...] = (".png",),
    device: Optional[torch.device] = None,
    data_range: Optional[float] = None,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_map = list_images(ref_dir, exts)
    pred_map = list_images(pred_dir, exts)

    common = sorted(set(ref_map.keys()) & set(pred_map.keys()))
    if not common:
        raise SystemExit("Tidak ada pasangan file yang cocok berdasarkan nama.")

    rows: List[Dict] = []
    n_ok = 0
    mse_all = []
    psnr_all = []

    print(f"Menilai {len(common)} file...")
    for name in common:
        ref_path = ref_map[name]
        pred_path = pred_map[name]
        try:
            ref = load_image_tensor(ref_path)
            pred = load_image_tensor(pred_path)

            # Validasi ukuran & kanal (tanpa resize)
            if ref.shape != pred.shape:
                raise ValueError(
                    f"Shape beda untuk {name}: ref {tuple(ref.shape)} vs pred {tuple(pred.shape)}"
                )

            # Pindah device
            ref_d = ref.to(device)
            pred_d = pred.to(device)

            # Data range otomatis dari referensi
            dr = _infer_data_range(ref_d) if data_range is None else float(data_range)

            # Hitung metrik
            mse_val = mse_per_image(ref_d, pred_d)[0].item()
            psnr_val = psnr_from_mse(torch.tensor([mse_val], device=device), dr)[0].item()

            h, w = ref.shape[1], ref.shape[2]
            c = ref.shape[0]
            rows.append({
                "filename": name,
                "width": w,
                "height": h,
                "channels": c,
                "mse": mse_val,
                "psnr": psnr_val,
            })
            mse_all.append(mse_val)
            psnr_all.append(psnr_val)
            n_ok += 1
        except Exception as e:
            # Catat error baris ini ke CSV juga (optional)
            rows.append({
                "filename": name,
                "width": "",
                "height": "",
                "channels": "",
                "mse": "",
                "psnr": "",
                "error": str(e),
            })
            print(f"[SKIP] {name}: {e}")

    # Tulis CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "width", "height", "channels", "mse", "psnr", "error"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if "error" not in r:
                r["error"] = ""
            w.writerow(r)

    if n_ok > 0:
        mse_mean = sum(mse_all) / n_ok
        psnr_mean = sum(psnr_all) / n_ok
        print(f"Selesai. Berhasil: {n_ok}/{len(common)}")
        print(f"Rata-rata MSE : {mse_mean:.6f}")
        print(f"Rata-rata PSNR: {psnr_mean:.6f} dB")
    else:
        print("Tidak ada pasangan yang valid untuk dihitung (semua gagal).")

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Hitung MSE & PSNR antar folder tanpa resize (PyTorch)."
    )
    ap.add_argument("--ref", type=Path, required=True, help="Folder ground-truth/reference")
    ap.add_argument("--pred", type=Path, required=True, help="Folder hasil/prediksi")
    ap.add_argument("--out", type=Path, required=True, help="File CSV output")
    ap.add_argument(
        "--exts",
        type=str,
        default=".png",
        help="Ekstensi yang dipakai, pisahkan dengan koma (mis: .png,.jpg). Default: .png",
    )
    ap.add_argument(
        "--data-range",
        type=float,
        default=None,
        help="Paksa data_range (mis: 1.0 untuk 0–1, 255.0 untuk 0–255). Default: auto.",
    )
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    exts = tuple([e.strip().lower() for e in args.exts.split(",") if e.strip()])
    evaluate_folders(
        ref_dir=args.ref,
        pred_dir=args.pred,
        out_csv=args.out,
        exts=exts,
        device=None,
        data_range=args.data_range,
    )
