import pandas as pd
import cv2
import numpy as np
import sys
#marker_list = ['DAPI', 'ACTG1','CD45','CD11B','COLLAGEN','CD20','PCNA','BCATENIN','PSTAT3','PEGFR','CGA','CD4','CD3D','HLAA','OLFM4','CD8','CD68','NAKATPASE','VIMENTIN','SOX9','FOXP3','LYSOZYME','SMA','ERBB2']
marker_list = ['DAPI', 'ACTG1','CD45','CD11B','COLLAGEN','CD20','PCNA','BCATENIN','PSTAT3','CGA','CD4','CD3D','HLAA','OLFM4','CD8','CD68','VIMENTIN','SOX9','FOXP3','LYSOZYME','SMA']



#df =pd.read_csv('/local_storage/baos1/ISBI24//Done/Q_SCORE/good_patch/min/post_manual_marker_QA_final.csv')
img_dir = '/local_storage/baos1/ISBI24/Done/AFRemoved'
tis_dir = '/local_storage/baos1/ISBI24/Done/TISSUE_MASK'



patch_size = 1024
#for index, row in df.iterrows():
tis_name = sys.argv[1] #row['Tissue_name']
h = int(sys.argv[2])#int(row['h'])
w = int(sys.argv[3])#int(row['w'])
#     print(index)
    
    
#img_tis_ret_mask = cv2.imread(f'{tis_dir}/{tis_name}_TISSUE_MASK.tif', cv2.IMREAD_GRAYSCALE)
img_tis_ret_mask = cv2.imread(f'{tis_dir}/{tis_name}_TISSUE_RETENTION.tif', cv2.IMREAD_GRAYSCALE)
print(f'{tis_dir}/{tis_name}_TISSUE_RETENTION.tif')
h_tis_ret, w_tis_ret = img_tis_ret_mask.shape
img_tis_ret_mask=img_tis_ret_mask[500:h_tis_ret, 0:w_tis_ret] # to delete description from the title.

    
img_tis_ret_mask_crop = img_tis_ret_mask[h*1024:(h+1)*1024, w*1024:(w+1)*1024]
img_tis_ret_mask_crop[img_tis_ret_mask_crop == 0] = 0
img_tis_ret_mask_crop[img_tis_ret_mask_crop == 255] = 1
    
img_list = []
for marker in marker_list:
        
    img = cv2.imread(f'{img_dir}/{tis_name}_{marker}.tif',cv2.IMREAD_GRAYSCALE)
        
    img_crop = img[h*1024:(h+1)*1024, w*1024:(w+1)*1024]
    img_crop_masked = cv2.bitwise_and(img_crop,img_crop,mask= img_tis_ret_mask_crop)    
    img_list.append(img_crop_masked)
        
img_stack = np.hstack(img_list)
cv2.imwrite(f'/local_storage/baos1/ISBI24/Done/bad_patch_train/{tis_name}_{marker}_{h}_{w}_masked.png', img_stack)
