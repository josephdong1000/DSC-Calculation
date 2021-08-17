#!/usr/bin/env python
import numpy as np
import nibabel as nib

# extract data array from files
def loaddata(pred, true):
    pred_path = "/gdrive/public/DATA/Human_Data/BIDS/derivatives" + pred
    true_path = "/gdrive/public/DATA/Human_Data/BIDS/derivatives" + true
    pred_img = nib.load(pred_path)
    true_img = nib.load(true_path)
    return [np.array(pred_img.dataobj), np.array(true_img.dataobj)]

# calculate Dice scores
def dice(pred_arr, true_arr, k = 1):
    intersection = np.sum(pred_arr[true_arr==k]) * 2.0
    dicescore = intersection / (np.sum(pred_arr) + np.sum(true_arr))
    return dicescore

# calculate directed Hausdorff distance
def dir_Hausdorff(vol_a,vol_b):
    dist_lst = []
    for idx in range(len(vol_a)):
        dist_min = 1000.0
        for idx2 in range(len(vol_b)):
            dist= np.linalg.norm(vol_a[idx]-vol_b[idx2])
            if dist_min > dist:
                dist_min = dist
        dist_lst.append(dist_min)
    return np.max(dist_lst)

# calculate symmetric Hausdorff distance
def hausdorff(a, b):
    return max(dir_Hausdorff(a, b), dir_Hausdorff(b, a))


# reading file paths
file = open('filepaths.txt', 'r')
filelines = file.readlines()
subjectID = []
sessionID = []

for i in filelines:
    i = i.strip()
    subjectID.append(i.split(" ")[0])
    sessionID.append(i.split(" ")[1])

for i in range(len(filelines)):
    RESSEG = "/Resseg/" + subjectID[i] + "/" + sessionID[i] + "/" + subjectID[i] + "_" + sessionID[
        i] + "_acq-3D_T1w_RESSEG.nii.gz"
    DR = "/DeepResection/" + subjectID[i] + "/" + sessionID[i] + "/" + subjectID[i] + "_predicted_mask.nii.gz"
    Manual = "/manualSegmentation/" + subjectID[i] + "/" + sessionID[i] + "/" + subjectID[i] + "_" + sessionID[
        i] + "_acq-3D_T1w.nii.gz"

    data_arr = loaddata(RESSEG, DR)

    pred_arr = data_arr[0]
    true_arr = data_arr[1]
    print("{}\t{}".format(dice(pred_arr, true_arr, k=1), hausdorff(pred_arr, true_arr)))
