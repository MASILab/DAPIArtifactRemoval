import pandas as pd
import cv2
df = pd.read_csv('/local_storage/baos1/ISBI24/Done/Q_SCORE/good_patch/min/final_combine_all_v3.csv')
img_dir = '/local_storage/baos1/ISBI24/Done/DAPI_ROUND'
tis_dir = '/local_storage/baos1/ISBI24/Done/TISSUE_MASK'



patch_size = 1024
for index, row in df.iterrows():
    tis_name = row['Tissue_name']
    print(index)
    
    img = cv2.imread(f'{img_dir}/{tis_name}_DAPI_DAPI_12ms_ROUND_00.tif',cv2.IMREAD_GRAYSCALE)
    img_tis_ret_mask = cv2.imread(f'{tis_dir}/{tis_name}_TISSUE_MASK.tif', cv2.IMREAD_GRAYSCALE)
    img_tis_ret_mask[img_tis_ret_mask == 0 ] = 0
    img_tis_ret_mask[img_tis_ret_mask == 255] = 1


    h = int(row['h'])
    w = int(row['w'])
    
    img_tis_ret_mask_crop = img_tis_ret_mask[h*1024:(h+1)*1024, w*1024:(w+1)*1024]
    img_crop = img[h*1024:(h+1)*1024, w*1024:(w+1)*1024]
    img_crop_masked = cv2.bitwise_and(img_crop,img_crop,mask= img_tis_ret_mask_crop)
    
    cv2.imwrite(f'/local_storage/baos1/ISBI24/Done/QA_1024/{tis_name}_DAPI_DAPI_12ms_ROUND_00_{h}_{w}_masked.png', img_crop_masked)

