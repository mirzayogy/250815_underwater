import pandas as pd # type: ignore
from sklearn.tree import DecisionTreeClassifier, plot_tree # type: ignore
from sklearn.model_selection import cross_val_score, StratifiedKFold #  type: ignore 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #   type: ignore
import matplotlib.pyplot as plt # type: ignore

df = pd.read_csv('tes02.csv')
print(df.columns)

try:
    # X = df.drop('result', axis=1) # Fitur
    X = df.drop(columns=['result','voh','ssim_b', 'width', 'height']) # Fitur

    y = df['result'] # Target
    print(f"\nKolom target diidentifikasi: 'result'")
except KeyError:
    print("Error: Kolbv dom target 'result' tidak ditemukan. Harap ganti dengan nama kolom target yang benar di CSV Anda.")
    exit()

model_dt_cv = DecisionTreeClassifier(random_state=42)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print(f"\nMelakukan pengujian Cross-Validation (dengan {skf.get_n_splits(X, y)} fold)...")

cv_scores = cross_val_score(model_dt_cv, X, y, cv=skf, scoring='accuracy')

print(f"Hasil Akurasi per fold: {cv_scores}")
print(f"Akurasi Rata-rata Cross-Validation: {cv_scores.mean():.4f}")
print(f"Standar Deviasi Akurasi Cross-Validation: {cv_scores.std():.4f}")

try:
    model_for_plot = DecisionTreeClassifier(random_state=42)
    model_for_plot.fit(X, y) # Melatih pada seluruh data untuk visualisasi

    # plt.figure(figsize=(20,15))
    # plot_tree(model_for_plot, feature_names=X.columns.tolist(), class_names=[str(c) for c in model_for_plot.classes_], filled=True, rounded=True)
    # plt.title("Visualisasi Decision Tree (Dilatih pada Seluruh Data)")
    # plt.savefig('decision_tree_cv.png')
    # print("\nVisualisasi Decision Tree disimpan sebagai 'decision_tree_cv.png'")
    plt.show()
except Exception as e:
    print(f"\nTidak dapat memvisualisasikan Decision Tree. Error: {e}")
    print("Pastikan semua fitur di X adalah numerik dan matplotlib sudah terinstal.")