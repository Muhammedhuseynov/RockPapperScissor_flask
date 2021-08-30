import numpy as np
import cv2
import os
from tqdm import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# for Imagenet Formats dataset's
# get Mean ,std



root = "./data/train"

def newCalc(data_dir):


    R_channel = 0
    G_channel = 0
    B_channel = 0
    img_size = 0

    # folders = {}
    folders = os.listdir(data_dir)

    get_locations = {}
    for foldName in folders:
        direc = os.path.join(data_dir,foldName)
        get_imgs = os.listdir(direc)
        get_locations[foldName] = get_imgs
    newImg = None
    all_imgs = []
    for foldname,imgs in get_locations.items():
        for img in get_locations[foldname]:
            img_dir = os.path.join(data_dir,foldname,img)

            all_imgs.append(img_dir)
            # print(foldname)



    for imgName in tqdm(all_imgs):

        img = cv2.imread(imgName)
        img = img / 255.0
        img_size = img_size + img.shape[0] * img.shape[1]
        R_channel = R_channel + np.sum(img[:, :, 0])
        G_channel = G_channel + np.sum(img[:, :, 1])
        B_channel = B_channel + np.sum(img[:, :, 2])

    R_mean = R_channel / img_size
    G_mean = G_channel / img_size
    B_mean = B_channel / img_size

    R_channel = 0
    G_channel = 0
    B_channel = 0
    for imgName in tqdm(all_imgs):
        img = cv2.imread(imgName)
        img = img / 255.0
        R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
        G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

    R_var = (R_channel / img_size) ** 0.5
    G_var = (G_channel / img_size) ** 0.5
    B_var = (B_channel / img_size) ** 0.5
    
    print([R_mean, G_mean, B_mean])
    print([R_var, G_var, B_var])
    



if __name__ == '__main__':
    newCalc(root)
    
# imageNet---
# R_mean is 0.394715, G_mean is 0.436628, B_mean is 0.428640
# R_var is 0.162872, G_var is 0.166768, B_var is 0.155795
# --------
# mean: [128.99203383, 108.45594001,  97.59257357]
# std: [67.37134603, 62.10296662, 61.10111998]
# [0.383155,0.425758,0.506294]
# [0.289657,0.290323,0.310512]
#
# R_mean is 0.383155, G_mean is 0.425758, B_mean is 0.506294
# R_var is 0.289657, G_var is 0.290323, B_var is 0.310512


