import numpy as np
import sys

qscore_dir = '/local_storage/baos1/ISBI24/Done/Q_SCORE/bad_patch'
img_prefix = sys.argv[1]

np_iou = np.load(f'{qscore_dir}/{img_prefix}_bad_patch_selection_IOU.npy')
min_np_iou = np.mean(np_iou, axis=2)

np_dice = np.load(f'{qscore_dir}/{img_prefix}_bad_patch_selection_DICE.npy')
min_np_dice = np.mean(np_dice, axis=2)

np.save(f'{qscore_dir}/mean/IOU/{img_prefix}_bad_patch_selection_IOU_mean.npy', min_np_iou)
np.save(f'{qscore_dir}/mean/DICE/{img_prefix}_bad_patch_selection_DICE_mean.npy', min_np_dice)
