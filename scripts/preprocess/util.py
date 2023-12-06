import numpy as np

def convert_to_binary_mask(mask, threshold=0):
    # Threshold the 32-bit mask to create a binary mask
    binary_mask = (mask > threshold).astype(np.uint8)
    return binary_mask
def calculate_dice_coefficient(binary_mask1, binary_mask2):
    # Ensure both binary masks have the same dimensions
    if binary_mask1.shape != binary_mask2.shape:
        raise ValueError("Both binary masks must have the same dimensions")

    # Calculate the intersection and union of the binary masks
    intersection = np.logical_and(binary_mask1, binary_mask2)
    union = np.logical_or(binary_mask1, binary_mask2)

    # Calculate the Dice coefficient
    dice_coefficient = 2 * np.sum(intersection) / (np.sum(binary_mask1) + np.sum(binary_mask2))

    return dice_coefficient
def calculate_iou_efficiently(mask1, mask2):
     # Ensure both masks have the same dimensions
     if mask1.shape != mask2.shape:
         raise ValueError("Both masks must have the same dimensions")
 
     # Calculate the intersection and union of the masks using NumPy operations
     intersection = np.logical_and(mask1, mask2)
     union = np.logical_or(mask1, mask2)
 
     # Sum the values in the intersection and union masks
     intersection_sum = np.sum(intersection)
     union_sum = np.sum(union)
 
     # Calculate IoU efficiently
     iou = intersection_sum / union_sum
 
     return iou 
