import cv2
from util import *
import math
import numpy as np
import sys
# get good region
img_prefix = sys.argv[1] # 'GCA081TIB_TISSUE02' #
dapi_rnd = 'DAPI_DAPI_12ms_ROUND'
nuc_dir = '/local_storage/baos1/ISBI24/Done/NUC'
tis_dir = '/local_storage/baos1/ISBI24/Done/TISSUE_MASK'
qscore_dir = '/local_storage/baos1/ISBI24/Done/Q_SCORE/bad_patch'
print('load all deepcell nuc image')
nuc_list = []
nuc_list.append(cv2.imread(f'{nuc_dir}/NUC_{img_prefix}_{dapi_rnd}_00.tif', cv2.IMREAD_UNCHANGED))

for idx in range(1,19,2):
    if idx < 10:
        idx = '0%s' % idx
    nuc_list.append(cv2.imread(f'{nuc_dir}/NUC_{img_prefix}_{dapi_rnd}_{idx}.tif', cv2.IMREAD_UNCHANGED))
    
#print('load tissue retention mask')
img_tis_ret_mask = cv2.imread(f'{tis_dir}/{img_prefix}_TISSUE_RETENTION.tif', cv2.IMREAD_GRAYSCALE)
#print(img_tis_ret_mask.shape)
h_tis_ret, w_tis_ret = img_tis_ret_mask.shape
img_tis_ret_mask=img_tis_ret_mask[500:h_tis_ret, 0:w_tis_ret] # to delete description from the title.

img_tis_ret_mask[img_tis_ret_mask == 0 ] = 0
img_tis_ret_mask[img_tis_ret_mask == 255] = 1

print(img_tis_ret_mask.shape)

#FOR BAD PATCH, NO NEED TO MASK
print('mask deepcell nuc image')
for idx in range(0,len(nuc_list)):
    print(nuc_list[idx].shape)
    nuc_list[idx] = cv2.bitwise_and(nuc_list[idx],nuc_list[idx],mask= img_tis_ret_mask)
    

# find the roundup w,h
h, w = img_tis_ret_mask.shape
patch_size = 1024
roundup_w = math.floor(w/patch_size)
roundup_h = math.floor(h/patch_size)

# 0 vs. (1,3,5,7,9,11,13,15,17)
np_iou = np.zeros((roundup_h,roundup_w, len(nuc_list)-1 )) # 00 vs. rest
np_dice = np.zeros((roundup_h,roundup_w, len(nuc_list)-1 ))

#mask_ratio = 0.5
mask_ratio = 0.25

for idx in range(1,len(nuc_list)):
    print('dice,iou on idx:%s' % idx)
    for i in range (0,roundup_w):
        for j in range (0,roundup_h):
            curX = 0 + i * patch_size
            curY = 0 + j * patch_size
            targetX = curX + patch_size
            targetY = curY + patch_size
            
            tmp_img_tis = img_tis_ret_mask[curY:targetY, curX:targetX] 
            # just count patch with larger 50% ratio -> 25%
            if np.count_nonzero(tmp_img_tis)/(patch_size*patch_size) > mask_ratio:
                tmp_img_00 = nuc_list[0][curY:targetY, curX:targetX]
                tmp_img_other = nuc_list[idx][curY:targetY, curX:targetX]

                tmp_iou = calculate_iou_efficiently(tmp_img_00,tmp_img_other)
                np_iou[j,i,idx -1 ] = tmp_iou

                tmp_img_00_binary = convert_to_binary_mask(tmp_img_00)
                tmp_img_01_binary = convert_to_binary_mask(tmp_img_other)

                tmp_dice = calculate_dice_coefficient(tmp_img_00_binary, tmp_img_01_binary)
                np_dice[j,i,idx -1 ] = tmp_dice

np.save(f'{qscore_dir}/{img_prefix}_bad_patch_selection_DICE.npy', np_dice)
np.save(f'{qscore_dir}/{img_prefix}_bad_patch_selection_IOU.npy', np_iou)
