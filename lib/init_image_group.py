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



def init_semua_raw():
    dataset_path = Path('UIEB/raw-890')
    file_path = list(dataset_path.glob(r'**/*.png'))
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], file_path))

    file_path = pd.Series(file_path).astype(str)
    labels = pd.Series(labels)

    df = pd.concat([file_path, labels], axis=1)
    df.columns = ['image', 'label']
    return df

def init_semua_png(path):
    dataset_path = Path(path)
    
    # Mencari semua file .png secara rekursif dalam path yang diberikan
    file_path = list(dataset_path.glob(r'**/*.png'))
    
    # --- PERUBAHAN DI SINI ---
    # Mengubah:
    # labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], file_path))
    # Menjadi:
    labels = list(map(lambda x: os.path.split(x)[1], file_path)) 
    # os.path.split(x) akan mengembalikan tuple (direktori_path, nama_file_lengkap)
    # [1] akan memilih nama_file_lengkap (misal: 'gambar_001.png')
    # -------------------------

    # Mengubah list ke pandas Series dan mengonversi path ke string
    file_path_series = pd.Series(file_path).astype(str)
    labels_series = pd.Series(labels)

    # Menggabungkan Series menjadi DataFrame
    df = pd.concat([file_path_series, labels_series], axis=1)
    df.columns = ['image', 'label'] # Memberi nama kolom
    return df

def show_image_group(image_group):
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20,10), subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        print(f"Showing image {i+1}/{len(image_group.image)}: {image_group.image[i]}")
        ax.imshow(plt.imread(image_group.image[i]))
        head, filename = os.path.split(image_group.image[i])
        ax.set_title(filename)
