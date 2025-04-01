import numpy as np
import pandas as pd
import openpyxl

file = '5ML-Riduzione-PCA-EsercizioGuidato-EsercizioAutonomia-Dati.xlsx'

# data = pd.read_excel(file, sheet_name=0, skiprows=2, usecols=(3,4,5,6,7,8,9), na_filter=False)
# data.dropna(how='all', inplace=True)
# print(data)

wb = openpyxl.load_workbook(file, data_only=True)
sheets = wb.sheetnames
page = 1
sheet = wb[sheets[page]]
print(sheets[page])

data = []
for row in sheet.iter_rows(values_only=True):
    first_not_none = -1
    last_not_none = -1
    for i, cell in enumerate(row):
        if cell is not None:
            if first_not_none == -1:
                first_not_none = i
            last_not_none = i

    if first_not_none != -1:
        data.append(row[first_not_none:last_not_none+1])

data = np.array(data)
print("\nPATTERN")
print(data)

corr = np.corrcoef(data, rowvar=False)
print("\nMATRICE DI CORRELAZIONE")
print(corr)

e_val, e_vec = np.linalg.eig(corr)

sort_index = e_val.argsort()[::-1]
e_val = e_val[sort_index]
e_vec = e_vec[sort_index, :]

#
print(sort_index)

e_val_sum = np.sum(e_val)

print("\nAUTOVETT MATRICE DI CORRELAZIONE")
print(e_vec)

print("\nAUTOVAL MATRICE DI CORRELAZIONE")
print(e_val)

val_sum = 0
print("\nPERCENTUALI")
for i, val in enumerate(e_val):
    val_sum += val
    print(round(val, 1), "\t", round(val/e_val_sum * 100, 1), "%\t", round(val_sum/e_val_sum * 100, 1), "%")

#print(e_val)
