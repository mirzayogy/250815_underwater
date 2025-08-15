import cv2# type: ignore
import numpy as np# type: ignore
import matplotlib.pyplot as plt# type: ignore

# Fungsi untuk menghitung delta
def compute_delta(hue, window_size=3):
    pad = window_size // 2
    h, w = hue.shape
    delta = np.zeros_like(hue, dtype=np.float32)
    
    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            # Ekstrak jendela tetangga ω(x, y)
            window = hue[y-pad:y+pad+1, x-pad:x+pad+1]
            center_value = hue[y, x]
            sum_neighbors = np.sum(window)
            K = window_size * window_size
            delta[y, x] = (sum_neighbors - center_value) / K
    return delta

# Fungsi untuk menghitung VOH
def compute_voh(delta):
    h, w = delta.shape
    return np.sum(delta) / (h * w)

# Baca gambar (ganti path jika diperlukan)
def compute_delta_and_voh(image_path, window_size=3):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32) / 179.0  # Normalisasi ke [0, 1]
    
    delta = compute_delta(hue, window_size)
    voh = compute_voh(delta)
    
    return voh

# # Tampilkan hasil
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.title('Hue')
# plt.imshow(hue, cmap='hsv')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('Delta(x, y)')
# plt.imshow(delta, cmap='jet')
# plt.colorbar()
# plt.axis('off')
# plt.show()

# # Cetak nilai VOH
# print(f"Nilai VOH (Variance of Hue): {voh:.6f}")


# ========== Example Usage ==========
# import lib.lib_gpt
# importlib.reload(lib.lib_gpt)
# from lib.lib_gpt import compute_delta_and_voh

# voh_gpt = []
# for idx, row in image_group.iterrows():
#     VOH_value = compute_delta_and_voh(row['image'], kernel_size)
#     voh_gpt.append(VOH_value)

# image_group['voh_gpt'] = voh_gpt