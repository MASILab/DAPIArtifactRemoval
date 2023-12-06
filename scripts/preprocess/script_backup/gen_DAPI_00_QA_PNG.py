import pandas as pd
import cv2
df = pd.read_csv('/local_storage/baos1/ISBI24/Done/Q_SCORE/good_patch/min/final_combine_all_v3.csv')
img_dir = '/local_storage/baos1/ISBI24/Done/'
patch_size = 1024
for index, row in df.iterrows():
    tis_name = row['Tissue_name']
    print(index)
    img = cv2.imread(f'{img_dir}/{tis_name}_DAPI_DAPI_12ms_ROUND_00.tif',cv2.IMREAD_GRAYSCALE)
    h = int(row['h'])
    w = int(row['w'])
    img_crop = img[h*1024:(h+1)*1024, w*1024:(w+1)*1024]
    cv2.imwrite(f'/local_storage/baos1/ISBI24/Done/QA_1024/{tis_name}_DAPI_DAPI_12ms_ROUND_00_{h}_{w}.png', img_crop)
