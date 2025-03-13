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



for d in data:
    print(d)