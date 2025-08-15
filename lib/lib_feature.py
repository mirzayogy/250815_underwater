import cv2 # type: ignore
import numpy as np # type: ignore
import cupy as cp # type: ignore
from skimage.metrics import structural_similarity as ssim # type: ignore
import cupyx.scipy.ndimage as cpx # type: ignore
import pandas as pd # type: ignore

def export_df(image_group):
    image_group = pd.concat([image_group, image_group['image'].apply(calculate_all_image_features)], axis=1)
    return image_group
   


def calculate_all_image_features(image_path):

    im_width, im_height = get_width_height(image_path)
    mean_r, mean_g, mean_b = get_mean(image_path)
    var_r, var_g, var_b = get_var(image_path)
    ssim_r, ssim_g, ssim_b = ssim_channel(image_path)
    mean_hue = get_mean_hue(image_path)
    voh = compute_delta_and_voh_cuda(image_path)

    return pd.Series({
        'width': im_width,
        'height': im_height,
        'mean_r': mean_r,
        'mean_g': mean_g,
        'mean_b': mean_b,
        'var_r': var_r,
        'var_g': var_g,
        'var_b': var_b,
        'ssim_r': ssim_r,
        'ssim_g': ssim_g,
        'ssim_b': ssim_b,
        'mean_hue': mean_hue,
        'voh': voh
    })

def check_image_cv(image):
  if(type(image) is np.ndarray):
    image_cv = image
  else:
    image_arr = np.asarray(image)
    b, g, r = cv2.split(image_arr)
    image_cv = cv2.merge([r, g, b])
  return image_cv

def ssim_channel(image_path):
    image_cv = cv2.imread(image_path)
    blue_channel = image_cv[:, :, 0]
    green_channel = image_cv[:, :, 1]
    red_channel = image_cv[:, :, 2]

    min_r, max_r = np.min(red_channel), np.max(red_channel)
    data_range = max_r - min_r

    green_channel_mean = np.mean(green_channel)
    blue_channel_mean = np.mean(blue_channel)

    if(blue_channel_mean > green_channel_mean):
      ssim_r= ssim(red_channel, blue_channel, data_range=data_range)
    else:
      ssim_r= ssim(red_channel, green_channel, data_range=data_range)

    ssim_g = ssim(green_channel, blue_channel, data_range=data_range)
    ssim_b = ssim_g

    return ssim_r, ssim_g, ssim_b

def get_mean(image_path):
    image_cv = cv2.imread(image_path)
    blue_channel = image_cv[:, :, 0]
    green_channel = image_cv[:, :, 1]
    red_channel = image_cv[:, :, 2]
    red_channel_mean = np.mean(red_channel)
    green_channel_mean = np.mean(green_channel)
    blue_channel_mean = np.mean(blue_channel)
    return red_channel_mean, green_channel_mean, blue_channel_mean

def get_var(image_path):
    image_cv = cv2.imread(image_path)
    blue_channel = image_cv[:, :, 0]
    green_channel = image_cv[:, :, 1]
    red_channel = image_cv[:, :, 2]
    red_channel_var = np.var(red_channel)
    green_channel_var = np.var(green_channel)
    blue_channel_var = np.var(blue_channel)
    return red_channel_var, green_channel_var, blue_channel_var

def get_width_height(image_path):
    image_cv = cv2.imread(image_path)
    height, width = image_cv.shape[:2]
    return width, height

def get_mean_hue(image_path):
    image_cv = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    hue_channel = img_hsv[:, :, 0].astype(np.float32)
    h_xy = hue_channel / 179.0
    mean_hue = np.mean(h_xy)
    # var_hue = np.var(h_xy)

    return mean_hue

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