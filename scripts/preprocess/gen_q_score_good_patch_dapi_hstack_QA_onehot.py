import cv2
from util import *
import math
import numpy as np
import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# get good region

prefix_list = ['GCA007ACB','GCA007TIB_TISSUE01','GCA007TIB_TISSUE02','GCA053ACB_TISSUE01','GCA053ACB_TISSUE02','GCA053TIA_TISSUE01','GCA053TIA_TISSUE02','GCA093ACB_TISSUE01','GCA093ACB_TISSUE02','GCA093TIA','GCA094ACA_TISSUE01','GCA094ACA_TISSUE02','GCA094TIB_TISSUE01','GCA094TIB_TISSUE02','GCA096ACB','GCA096TIB','GCA099TIA','GCA112ACB','GCA112TIA','GCA113ACA','GCA113TIA','GCA118ACB_TISSUE01','GCA118ACB_TISSUE02','GCA118TIA_TISSUE01','GCA118TIA_TISSUE02','GCA132ACB_TISSUE01','GCA132ACB_TISSUE02','GCA132ACB_TISSUE03','GCA132TIA_TISSUE01','GCA132TIA_TISSUE02','GCA132TIA_TISSUE03']
set01_list = ['GCA002ACB','GCA002TIB','GCA003ACA','GCA003TIB','GCA004TIB','GCA011ACB','GCA011TIB','GCA012ACB','GCA012TIB']

img_prefix = sys.argv[1] # 'GCA081TIB_TISSUE02' #

if img_prefix in prefix_list:
    dapi_rnd_00 = 'DAPI_DAPI_30ms_ROUND'
    dapi_rnd_01 = 'DAPI_DAPI_14ms_ROUND'
elif img_prefix in set01_list:
    dapi_rnd_00 = 'DAPI_UV_12ms_ROUND'
    dapi_rnd_01 = dapi_rnd_00
else:
    dapi_rnd_00 = 'DAPI_DAPI_12ms_ROUND'
    dapi_rnd_01 = dapi_rnd_00
    
nuc_dir = '/local_storage/baos1/ISBI24/Done/NUC'
tis_dir = '/local_storage/baos1/ISBI24/Done/TISSUE_MASK'
qscore_dir = '/local_storage/baos1/ISBI24/Done/Q_SCORE/onehot/'

print('load all deepcell nuc image')
nuc_list = []
nuc_list.append(np.array(Image.open(f'{nuc_dir}/NUC_{img_prefix}_{dapi_rnd_00}_00.tif')))
#nuc_list.append(np.array(Image.open(f'{nuc_dir}/NUC_{img_prefix}_{dapi_rnd_01}_01.tif')))
    
print('load tissue retention mask')
img_tis_ret_mask = cv2.imread(f'{tis_dir}/{img_prefix}_TISSUE_MASK.tif', cv2.IMREAD_GRAYSCALE)
img_tis_ret_mask[img_tis_ret_mask == 0 ] = 0
img_tis_ret_mask[img_tis_ret_mask == 255] = 1

print('mask deepcell nuc image')
for idx in range(0,len(nuc_list)):
    nuc_list[idx] = cv2.bitwise_and(nuc_list[idx],nuc_list[idx],mask= img_tis_ret_mask)


# create patch selection for QA
dapi_dir = '/local_storage/baos1/ISBI24/Done/AFRemoved/'
dapi_round_dir = '/local_storage/baos1/ISBI24/Done/DAPI_ROUND/'
dapi_list = []
dapi_list.append(cv2.imread(f'{dapi_dir}/{img_prefix}_DAPI.tif',cv2.IMREAD_GRAYSCALE))
#dapi_list.append(cv2.imread(f'{dapi_round_dir}/{img_prefix}_{dapi_rnd_01}_01.tif',cv2.IMREAD_GRAYSCALE))

for idx in range(0,len(dapi_list)):
    dapi_list[idx] = cv2.bitwise_and(dapi_list[idx],dapi_list[idx],mask= img_tis_ret_mask)
    
# find the roundup w,h
h, w = img_tis_ret_mask.shape
patch_size = 1024
roundup_w = math.floor(w/patch_size)
roundup_h = math.floor(h/patch_size)

# 0 vs. (1,3,5,7,9,11,13,15,17) -> 0 vs. 1
# np_iou = np.zeros((roundup_h,roundup_w, len(nuc_list)-1 )) # 00 vs. rest
# np_dice = np.zeros((roundup_h,roundup_w, len(nuc_list)-1 ))

#np_iou = np.zeros((roundup_h,roundup_w)) # 00 vs. 01
#np_dice = np.zeros((roundup_h,roundup_w ))

mask_ratio = 0.5


for i in range (0,roundup_w):
    for j in range (0,roundup_h):
        curX = 0 + i * patch_size
        curY = 0 + j * patch_size
        targetX = curX + patch_size
        targetY = curY + patch_size
            
        tmp_img_tis = img_tis_ret_mask[curY:targetY, curX:targetX] 
        # just count patch with larger 50% ratio
        if np.count_nonzero(tmp_img_tis)/(patch_size*patch_size) > mask_ratio:
            tmp_dapi_list = []
            tmp_dapi_list.append(dapi_list[0][curY:targetY, curX:targetX])
        
            tmp_img_00 = nuc_list[0][curY:targetY, curX:targetX]
#            tmp_img_other = nuc_list[1][curY:targetY, curX:targetX] # idx is 1 since only 2 element is available

#            tmp_iou = calculate_iou_efficiently(tmp_img_00,tmp_img_other)
#            np_iou[j,i] = tmp_iou

            tmp_img_00_binary = convert_to_binary_mask(tmp_img_00)
            tmp_dapi_list.append(tmp_img_00_binary)
            print(np.unique(tmp_img_00_binary))
            #print(tmp_img_00_binary.dtype)
           
#            tmp_img_01_binary = convert_to_binary_mask(tmp_img_other)

#            tmp_dice = calculate_dice_coefficient(tmp_img_00_binary, tmp_img_01_binary)
#            np_dice[j,i] = tmp_dice
            tmp_img_00_binary_v2 = convert_to_binary_mask(tmp_img_00)
            tmp_img_00_binary_v2[tmp_img_00_binary_v2 > 0] = 255
            # create DAPI vis QA
          #  tmp_dapi_list.append(dapi_list[0][curY:targetY, curX:targetX])
#            tmp_dapi_list.append(dapi_list[1][curY:targetY, curX:targetX])
            tmp_dapi_list.append(tmp_img_00_binary_v2)

            
            tmp_dapi_merge = np.hstack(tmp_dapi_list)
            cv2.imwrite(f'{qscore_dir}/img_QA/{img_prefix}_DAPI_vis_QA_{j}_{i}.png', tmp_dapi_merge)
            
#np.save(f'{qscore_dir}/{img_prefix}_good_patch_selection_DICE.npy', np_dice)
#np.save(f'{qscore_dir}/{img_prefix}_good_patch_selection_IOU.npy', np_iou)





