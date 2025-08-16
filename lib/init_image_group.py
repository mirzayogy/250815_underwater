from pathlib import Path
import os
import pandas as pd # type: ignore
from matplotlib import pyplot as plt # type: ignore

def init_pernah(type='raw-890'):
    im_pernah = []
    im_pernah.append("UIEB/"+type+"/10151.png")
    im_pernah.append("UIEB/"+type+"/112_img_.png")
    im_pernah.append("UIEB/"+type+"/121_img_.png")
    im_pernah.append("UIEB/"+type+"/142_img_.png")
    im_pernah.append("UIEB/"+type+"/18_img_.png")
    im_pernah.append("UIEB/"+type+"/202_img_.png")
    im_pernah.append("UIEB/"+type+"/334_img_.png")
    im_pernah.append("UIEB/"+type+"/342_img_.png")
    im_pernah.append("UIEB/"+type+"/383_img_.png")
    im_pernah.append("UIEB/"+type+"/442_img_.png")
    im_pernah.append("UIEB/"+type+"/471_img_.png")
    im_pernah.append("UIEB/"+type+"/486_img_.png")
    im_pernah.append("UIEB/"+type+"/504_img_.png")
    im_pernah.append("UIEB/"+type+"/515_img_.png")
    im_pernah.append("UIEB/"+type+"/57_img_.png")
    im_pernah.append("UIEB/"+type+"/702_img_.png")
    im_pernah.append("UIEB/"+type+"/747_img_.png")
    im_pernah.append("UIEB/"+type+"/86_img_.png")
    im_pernah.append("UIEB/"+type+"/8_img_.png")
    im_pernah.append("UIEB/"+type+"/906_img_.png")

    im_filename = []
    for im in im_pernah:
        head, filename = os.path.split(im)
        im_filename.append(filename)

    dict_combo = {'image':im_pernah,'label': im_filename}
    df = pd.DataFrame(dict_combo)

    return df

def init_pilihan(type='raw-890'):
    im_pilihan = ["12290.png",
        "12324.png",
        "12348.png",
        "716_img_.png",
        "455_img_.png",
        "265_img_.png",
        "670_img_.png",
        "415_img_.png",
        "768_img_.png",
        "601_img_.png",
        "404_img_.png",
        "820_img_.png",
        "916_img_.png",
        "690_img_.png",
        "666_img_.png",
        "11052.png",
        "12445.png",
        "389_img_.png",
        "563.png",
        "10151.png",
        "613_img_.png",
        "567_img_.png",
        "5818.png",
        "841_img_.png",
        "2787.png",
        "15001.png",
        "99_img_.png",
        "244_img_.png",
        "332_img_.png",
        "700_img_.png",
        "413_img_.png",
        "337_img_.png",
        "365_img_.png",
        "722_img_.png",
        "62_img_.png",
        "469_img_.png",
        "921_img_.png",
        "381_img_.png",
        "240_img_.png",
        "271_img_.png",
        "29_img_.png",
        "392_img_.png",
        "78_img_.png",
        "360_img_.png",
        "405_img_.png",
        "757_img_.png",
        "202_img_.png",
        "851_img_.png",
        "230_img_.png",
        "3650.png"]
    
    im_pilihan_path = [f"../UIEB/raw-890/{nama}" for nama in im_pilihan]
    im_pilihan_ref_path = [f"../UIEB/reference-890/{nama}" for nama in im_pilihan]
        
    
    return pd.DataFrame({'image': im_pilihan_path, 'label': im_pilihan, 'ref': im_pilihan_ref_path})
    



def init_semua_raw():
    dataset_path = Path('../UIEB/raw-890')
    file_path = list(dataset_path.glob(r'**/*.png'))
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], file_path))

    file_path = pd.Series(file_path).astype(str)
    labels = pd.Series(labels)

    df = pd.concat([file_path, labels], axis=1)
    df.columns = ['image', 'label']
    return df

def init_semua_png(path):
    dataset_path = Path(path)
    
    # Cari semua file .png secara rekursif
    file_path = list(dataset_path.glob('**/*.png'))
    
    # Ambil nama file sebagai label
    labels = list(map(lambda x: os.path.split(x)[1], file_path)) 
    
    # Series path asli dan label
    file_path_series = pd.Series(file_path).astype(str)
    labels_series = pd.Series(labels)

    # Buat kolom ref: ganti 'raw' dengan 'reference'
    ref_series = file_path_series.str.replace("raw", "reference", regex=False)

    # Gabungkan ke DataFrame
    df = pd.concat([file_path_series, labels_series, ref_series], axis=1)
    df.columns = ['image', 'label', 'ref']

    return df

def show_image_group(image_group):
    fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(20,20), subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        # print(f"Showing image {i+1}/{len(image_group.image)}: {image_group.image[i]}")
        ax.imshow(plt.imread(image_group.image[i]))
        head, filename = os.path.split(image_group.image[i])
        ax.set_title(filename)

def get_filenames_with_prefix(csv_path, cluster_value=2, prefix="../UIEB/raw-890/"):
    df = pd.read_csv(csv_path)
    out = (
        df.loc[df["cluster_kmeans"] == cluster_value, ["filename"]]
          .copy()
          .reset_index(drop=True)
          .assign(filename=lambda d: prefix + d["filename"])
    )
    return out

def get_filenames(csv_path, cluster_value=2):
    df = pd.read_csv(csv_path)
    out = (
        df.loc[df["cluster_kmeans"] == cluster_value, ["filename"]]
          .copy()
          .reset_index(drop=True)
          .assign(filename=lambda d: d["filename"])
    )
    return out

import os
import math
import matplotlib.pyplot as plt
from matplotlib.image import imread

def show_image_group_auto(image_group, max_cols=6, cell_size=(3, 3), hide_empty=True):
    """
    Tampilkan grid gambar dengan ukuran grid otomatis.
    
    Params
    ------
    image_group : object dengan atribut .image berupa list path gambar
    max_cols    : kolom maksimum yang dipakai (default 6)
    cell_size   : (width, height) inch per sel (default (3,3))
    hide_empty  : jika True, axes kosong dihapus (lebih rapi)
    """
    # Ambil daftar path gambar
    paths = list(getattr(image_group, "image", []))
    n = len(paths)

    if n == 0:
        print("Tidak ada gambar untuk ditampilkan.")
        return

    # Hitung kolom & baris otomatis
    cols = min(max_cols, max(1, math.ceil(math.sqrt(n))))
    rows = math.ceil(n / cols)

    # Figure size proporsional jumlah sel
    fig_w = max(1, cols * cell_size[0])
    fig_h = max(1, rows * cell_size[1])

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_w, fig_h),
                             subplot_kw={'xticks':[], 'yticks':[]})
    # Pastikan axes flattened (kompatibel untuk kasus rows/cols == 1)
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    # Render tiap gambar
    shown = 0
    for i, ax in enumerate(axes):
        if i >= n:
            # Habis gambar
            if hide_empty:
                fig.delaxes(ax)
            continue

        img_path = paths[i]
        try:
            ax.imshow(imread(img_path))
            _, filename = os.path.split(img_path)
            ax.set_title(filename)
            shown += 1
        except Exception as e:
            # Gagal baca gambar: kosongkan axes ini
            ax.text(0.5, 0.5, f"Gagal baca:\n{os.path.basename(img_path)}",
                    ha='center', va='center', fontsize=8)
            ax.set_frame_on(True)

    plt.tight_layout()
    plt.show()

    # Opsional: return fig/axes kalau mau dipakai lanjut
    return fig, axes

