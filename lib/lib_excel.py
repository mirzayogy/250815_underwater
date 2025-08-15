import pandas as pd  # type: ignore
def export_to_excel(image_group, filename='image_group.xlsx'):
    with pd.ExcelWriter(filename) as writer:
        image_group.to_excel(writer, index=False, sheet_name='ImageGroup')