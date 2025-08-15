import cv2 # type: ignore
import numpy as np # type: ignore

def calculate_delta_hue(image_path, kernel_size=3):
    """
    Menghitung delta(x, y) untuk setiap piksel berdasarkan formula hue neighborhood.

    Args:
        image_path (str): Jalur menuju file gambar.
        kernel_size (int): Ukuran kernel untuk wilayah tetangga (e.g., 3 untuk 3x3).

    Returns:
        np.ndarray: Gambar hasil delta(x,y) (perbedaan hue lokal).
    """

    # 1. Muat gambar
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Error: Tidak dapat memuat gambar dari {image_path}. Pastikan jalur file benar.")
            return None
    except Exception as e:
        print(f"Error saat memuat gambar: {e}")
        return None

    # 2. Konversi ke HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 3. Ekstrak kanal Hue (H) dan normalisasi ke [0, 1]
    # Hue OpenCV [0, 179]. Normalisasi ke [0, 1]
    hue_channel = img_hsv[:, :, 0].astype(np.float32)
    h_xy = hue_channel / 179.0

    # 4. Tentukan ukuran wilayah tetangga K
    K = kernel_size * kernel_size

    # 5. Hitung jumlah Hue di wilayah tetangga (h_sum)
    # Gunakan cv2.blur() untuk mendapatkan rata-rata, lalu kalikan dengan K untuk mendapatkan jumlahnya.
    h_mean = cv2.blur(h_xy, (kernel_size, kernel_size))
    h_sum = h_mean * K

    # 6. Terapkan formula delta(x, y) = (h_sum - h(x, y)) / K
    delta_xy = (h_sum - h_xy) / K

    return delta_xy

def calculate_VOH(delta_result_array):
    """
    Menghitung VOH (nilai rata-rata dari delta(x, y) di seluruh gambar).

    Args:
        delta_result_array (np.ndarray): Array 2D yang berisi nilai delta(x, y) 
                                         untuk setiap piksel.

    Returns:
        float: Nilai VOH.
    """
    
    # Dapatkan M x N (total jumlah piksel)
    # NumPy menyediakan atribut .size untuk jumlah total elemen dalam array
    MN = delta_result_array.size
    
    # Hitung jumlah dari semua delta(x, y) (Sigma(x,y) delta(x,y))
    sum_delta = np.sum(delta_result_array)
    
    # Hitung VOH = (1 / MN) * Sum(delta)
    VOH = sum_delta / MN
    
    # Alternatif yang lebih sederhana dan lebih efisien di NumPy:
    # VOH = np.mean(delta_result_array)
    
    return VOH

# --- Contoh Penggunaan di Jupyter Notebook ---
# import lib.lib_gmn
# importlib.reload(lib.lib_gmn)
# from lib.lib_gmn import calculate_delta_hue, calculate_VOH

# voh_gmn = []
# for idx, row in image_group.iterrows():
#     delta_result = calculate_delta_hue(row['image'], kernel_size)
#     VOH_value = calculate_VOH(delta_result)
#     voh_gmn.append(VOH_value)

# image_group['voh_gmn'] = voh_gmn


def resize_image_opencv(image_path, output_path, new_width=None, new_height=None, scale_factor=None, interpolation=cv2.INTER_AREA):
    """
    Mengubah ukuran gambar menggunakan OpenCV.

    Args:
        image_path (str): Jalur ke file gambar input.
        output_path (str): Jalur untuk menyimpan gambar hasil resize.
        new_width (int, optional): Lebar baru dalam piksel.
        new_height (int, optional): Tinggi baru dalam piksel.
        scale_factor (float, optional): Faktor skala (misal, 0.5 untuk setengah ukuran).
        interpolation (int, optional): Metode interpolasi (e.g., cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC).
                                       cv2.INTER_AREA direkomendasikan untuk mengecilkan, 
                                       cv2.INTER_LINEAR/cv2.INTER_CUBIC untuk membesarkan.
    Returns:
        bool: True jika berhasil, False jika gagal.
    """
    try:
        # Muat gambar
        img = cv2.imread(image_path)

        if img is None:
            print(f"Error: Tidak dapat memuat gambar dari {image_path}. Pastikan jalur file benar.")
            return False

        original_height, original_width = img.shape[:2]
        
        if scale_factor:
            # Mengubah ukuran berdasarkan faktor skala
            resized_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=interpolation)
            # print(f"Ukuran asli: {original_width}x{original_height} px")
            # print(f"Ukuran baru (faktor skala {scale_factor}): {resized_img.shape[1]}x{resized_img.shape[0]} px")
        elif new_width and new_height:
            # Mengubah ukuran ke lebar dan tinggi tertentu
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
            # print(f"Ukuran asli: {original_width}x{original_height} px")
            # print(f"Ukuran baru: {new_width}x{new_height} px")
        elif new_width:
            # Mengubah ukuran berdasarkan lebar baru, mempertahankan aspek rasio
            aspect_ratio = original_height / original_width
            calculated_height = int(new_width * aspect_ratio)
            resized_img = cv2.resize(img, (new_width, calculated_height), interpolation=interpolation)
            # print(f"Ukuran asli: {original_width}x{original_height} px")
            # print(f"Ukuran baru: {new_width}x{calculated_height} px (aspek rasio dipertahankan)")
        elif new_height:
            # Mengubah ukuran berdasarkan tinggi baru, mempertahankan aspek rasio
            aspect_ratio = original_width / original_height
            calculated_width = int(new_height * aspect_ratio)
            resized_img = cv2.resize(img, (calculated_width, new_height), interpolation=interpolation)
            # print(f"Ukuran asli: {original_width}x{original_height} px")
            # print(f"Ukuran baru: {calculated_width}x{new_height} px (aspek rasio dipertahankan)")
        else:
            print("Error: Harap tentukan new_width, new_height, atau scale_factor.")
            return False

        # Simpan gambar yang sudah di-resize
        cv2.imwrite(output_path, resized_img)
        # print(f"Gambar berhasil disimpan ke: {output_path}")

        return True, original_width, original_height

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        return False, 0, 0