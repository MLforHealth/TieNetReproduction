import numpy as np
import dicom
import pandas as pd
import os

image_files = os.listdir('/crimea/mimic-cxr/_images/')
dataframe = pd.read_csv("/crimea/mimic-cxr/mimic-cxr-map.csv")
temp = []
for idx, row in dataframe.iterrows():
    if (str(row['dicom_id']) + '.dcm') in image_files:
         plan = dicom.read_file(str(row['dicom_id']) + '.dcm', stop_before_pixels=False)
         temp.append(plan.ViewPosition)
dataframe['view_position'] = temp
dataframe.to_csv("/crimea/liuguanx/new_map.csv")