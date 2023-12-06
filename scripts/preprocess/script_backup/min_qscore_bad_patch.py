import numpy as np
import sys

qscore_dir = '/local_storage/baos1/ISBI24/Done/Q_SCORE/bad_patch'
img_prefix = sys.argv[1]

np_iou = np.load(f'{qscore_dir}/{img_prefix}_bad_patch_selection_IOU.npy')
min_np_iou = np.min(np_iou, axis=2)

np_dice = np.load(f'{qscore_dir}/{img_prefix}_bad_patch_selection_DICE.npy')
min_np_dice = np.min(np_dice, axis=2)

np.save(f'{qscore_dir}/min/IOU/{img_prefix}_bad_patch_selection_IOU_min.npy', min_np_iou)
np.save(f'{qscore_dir}/min/DICE/{img_prefix}_bad_patch_selection_DICE_min.npy', min_np_dice)
