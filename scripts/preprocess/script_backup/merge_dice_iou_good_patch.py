import numpy as np
import sys
import pandas as pd

img_prefix = sys.argv[1]
qscore_dir = '/local_storage/baos1/ISBI24/Done/Q_SCORE/good_patch/min/'

dice_min = np.load(f'{qscore_dir}/DICE/{img_prefix}_good_patch_selection_DICE_min.npy')
iou_min = np.load(f'{qscore_dir}/IOU/{img_prefix}_good_patch_selection_IOU_min.npy')

num_rows, num_cols = dice_min.shape

data_dice_min = []
for h in range(num_rows):
    for w in range(num_cols):
        data_dice_min.append([dice_min[h, w], h, w])
df_dice_min = pd.DataFrame(data_dice_min, columns=['DICE_min', 'h', 'w'])

data_iou_min = []
for h in range(num_rows):
    for w in range(num_cols):
        data_iou_min.append([iou_min[h, w], h, w])
df_iou_min = pd.DataFrame(data_iou_min, columns=['IOU_min', 'h', 'w'])

merged_df = pd.merge(df_dice_min, df_iou_min, on=['h', 'w'], how='inner')

sorted_df = merged_df.sort_values(by=['DICE_min', 'IOU_min'], ascending=[False, False])
sorted_df['Tissue_name'] = img_prefix

sorted_df.to_csv(f'{qscore_dir}/{img_prefix}_good_patch_selection_final.csv')

