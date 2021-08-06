#!/usr/bin/env python

# Dice similarity function
import numpy as np
import nibabel as nib
import sys

# take in predicted and true paths from cmd line
pred_path = sys.argv[1]
true_path = sys.argv[2]

# convert the paths to img objects
pred_img = nib.load(pred_path)
true_img = nib.load(true_path)

# extract data array
pred_arr = np.array(pred_img.dataobj)
true_arr = np.array(true_img.dataobj)

# calculate dice scores
def dice(pred_arr, true_arr, k = 1):
    intersection = np.sum(pred_arr[true_arr==k]) * 2.0
    dicescore = intersection / (np.sum(pred_arr) + np.sum(true_arr))
    return dicescore

print(dice(pred_arr, true_arr, k = 1))
