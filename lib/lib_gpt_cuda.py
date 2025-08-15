import cv2 # type: ignore
import cupy as cp # type: ignore
import cupyx.scipy.ndimage as cpx # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

# Baca gambar
def compute_delta_and_voh_cuda(image_path, kernel_size=3):
    bgr = cv2.imread(image_path)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue_cpu = hsv[:, :, 0].astype(np.float32) / 179.0  # Normalisasi

    # Pindahkan ke GPU
    hue = cp.asarray(hue_cpu)

    # Hitung sum dari neighborhood dengan filter rata-rata (contoh 3x3 kernel)
    K = kernel_size * kernel_size
    kernel = cp.ones((kernel_size, kernel_size), dtype=cp.float32)

    # Konvolusi untuk menghitung jumlah tetangga
    sum_neighbors = cpx.convolve(hue, kernel, mode='constant', cval=0.0)

    # Hitung delta(x, y)
    delta = (sum_neighbors - hue) / K

    # Hitung VOH
    voh = cp.mean(delta).get()
    return voh

# # Pindahkan kembali ke CPU untuk visualisasi
# delta_cpu = cp.asnumpy(delta)

# # Tampilkan
# plt.imshow(delta_cpu, cmap='jet')
# plt.title(f'Delta(x, y) - VOH: {float(voh):.6f}')
# plt.colorbar()
# plt.axis('off')
# plt.show()