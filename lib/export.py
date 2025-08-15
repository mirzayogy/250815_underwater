from pathlib import Path
import pandas as pd # type: ignore

def df_to_csv(df, filename):
    out_csv = Path("csv/"+filename+".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("DataFrame berhasil dikonversi menjadi CSV!")


def df_columns_to_csv(kolom, results, filename):
    out_csv = Path("csv/"+filename+".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results, columns=kolom)
    df.to_csv(out_csv, index=False)
    print("DataFrame berhasil dikonversi menjadi CSV!")
