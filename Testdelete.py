import pandas as pd
import numpy as np
import os
Path = 'C:/Users/rocky/Desktop/Not/'
file = 'try.xls'
Workbook = pd.read_excel(os.path.join(Path + file), sheet_name='Summary', header=1)
Path = 'C:/Users/rocky/Desktop/Delete/'
file = 'DO NOT SAVE - Matlab Excel Template V2.xlsm'
Workbook.to_excel(os.path.join(Path + file), sheet_name='Data Extract Input', startrow=2, engine=io.excel.xlsm.writer)
print(Workbook)