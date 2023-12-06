import pandas as pd
import cv2
import numpy as np
import sys
marker_list = ['MUC2','ACTG1','CD45','CD11B','COLLAGEN','CD20','PCNA','BCATENIN','PSTAT3','PEGFR','CGA','CD4','COX2','CD3D','HLAA','PANCK','OLFM4','CD8','ACTININ','CD68','NAKATPASE','VIMENTIN','SOX9','FOXP3','LYSOZYME','SMA','ERBB2']



df =pd.read_csv('dapi_round_final/remained_qa_marker_v2.csv')
img_dir = '/local_storage/baos1/ISBI24/Done/AFRemoved'
tis_dir = '/local_storage/baos1/ISBI24/Done/TISSUE_MASK'



patch_size = 1024
for index, row in df.iterrows():
    tis_name = row['Tissue_name']
    h = int(row['h'])
    w = int(row['w'])
    print(index)
    
    
    img_tis_ret_mask = cv2.imread(f'{tis_dir}/{tis_name}_TISSUE_MASK.tif', cv2.IMREAD_GRAYSCALE)
    
    img_tis_ret_mask_crop = img_tis_ret_mask[h*1024:(h+1)*1024, w*1024:(w+1)*1024]
    img_tis_ret_mask_crop[img_tis_ret_mask_crop == 0] = 0
    img_tis_ret_mask_crop[img_tis_ret_mask_crop == 255] = 1


    
    
    for marker in marker_list:
        dapi = cv2.imread(f'{img_dir}/{tis_name}_DAPI.tif',cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(f'{img_dir}/{tis_name}_{marker}.tif',cv2.IMREAD_GRAYSCALE)
        
        img_crop = img[h*1024:(h+1)*1024, w*1024:(w+1)*1024]
        img_crop_masked = cv2.bitwise_and(img_crop,img_crop,mask= img_tis_ret_mask_crop)
        
        dapi_crop = dapi[h*1024:(h+1)*1024, w*1024:(w+1)*1024]
        dapi_crop_masked = cv2.bitwise_and(dapi_crop,dapi_crop,mask= img_tis_ret_mask_crop)
        
        img_list = []
        img_list.append(dapi_crop_masked)
        img_list.append(img_crop_masked)
        
        img_stack = np.hstack(img_list)
    
        cv2.imwrite(f'/local_storage/baos1/ISBI24/Done/QA_1024/AFRemoved_final//{tis_name}_{marker}_{h}_{w}_masked.png', img_stack)

