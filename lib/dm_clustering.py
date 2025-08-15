import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.cluster import KMeans # type: ignore
from sklearn.neighbors import kneighbors_graph # type: ignore
from sklearn.cluster import SpectralClustering # type: ignore

def kmeans_from_df(
    df: pd.DataFrame,
    n_clusters: int = 3,
    features: list | None = None,
    standardize: bool = True,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300
):
    """
    K-Means dari input DataFrame.

    Param:
      df           : DataFrame input.
      n_clusters   : jumlah klaster.
      features     : list kolom fitur; jika None pakai kolom numerik saja.
      standardize  : apakah fitur di-Standardize (mean=0, std=1).
      random_state : seed untuk reprodusibilitas.
      n_init       : jumlah inisialisasi KMeans.
      max_iter     : iterasi maksimum KMeans.

    Return:
      labels (np.ndarray), model (KMeans), X_used (np.ndarray), cols (list)
    """
    # Pilih kolom fitur
    if features is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = features

    if not cols:
        raise ValueError("Tidak ada kolom numerik yang ditemukan/ditentukan untuk klastering.")

    # Buang baris yang ada NaN pada fitur
    X_df = df[cols].dropna().copy()
    X_used = X_df.to_numpy()

    # Standarisasi (opsional)
    if standardize:
        scaler = StandardScaler()
        X_used = scaler.fit_transform(X_used)

    # Model KMeans
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter
    )
    labels = km.fit_predict(X_used)

    return labels, km, X_used, cols



def knn_clustering_from_df(
    df: pd.DataFrame,
    n_clusters: int = 3,
    k_neighbors: int = 10,
    features: list | None = None,
    standardize: bool = True,
    random_state: int = 42
):
    """
    Klastering berbasis k-NN (graf k tetangga) dengan Spectral Clustering.

    Param:
      df            : DataFrame input.
      n_clusters    : jumlah klaster yang diinginkan.
      k_neighbors   : jumlah tetangga (k) untuk membangun graf k-NN.
      features      : daftar kolom fitur; jika None akan pakai kolom numerik saja.
      standardize   : apakah fitur dinormalisasi (StandardScaler).
      random_state  : seed untuk reprodusibilitas.

    Return:
      labels (np.ndarray), model (SpectralClustering), X_used (np.ndarray), cols (list)
    """
    # Pilih fitur
    if features is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = features
    if not cols:
        raise ValueError("Tidak ada kolom numerik yang ditemukan/ditentukan untuk klastering.")

    # Ambil data dan tangani NaN sederhana (drop baris yang punya NaN pada fitur)
    X_used = df[cols].dropna().to_numpy()

    # Standarisasi (opsional)
    if standardize:
        X_used = StandardScaler().fit_transform(X_used)

    # Bangun graf k-NN (sparse matrix)
    knn_graph = kneighbors_graph(
        X_used,
        n_neighbors=k_neighbors,
        mode="connectivity",   # atau "distance"
        include_self=False,
        n_jobs=-1
    )

    # Spectral Clustering dengan affinity dari graf k-NN
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed_nearest_neighbors",
        n_neighbors=k_neighbors,
        assign_labels="kmeans",
        random_state=random_state,
        n_init=10
    )
    labels = model.fit_predict(X_used)

    return labels, model, X_used, cols

