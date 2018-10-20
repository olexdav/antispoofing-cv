# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 13:09:52 2018

@author: olexd
"""

import os
import cv2
import random

inputFolder = "spoofs_raw"
outputFolder = "spoofs"

# crop image by X% from each side
def crop_around(image, percentage):
    h, w = image.shape[:2]
    offx, offy = int(w * percentage), int(h * percentage)
    return image[offy:h-offy, offx:w-offx, :]

# crop the image so it becomes square
def crop_square(image):
    h, w = image.shape[:2]
    if h > w:
        offset = int((h-w)/2)
        return image[offset:h-offset, :, :]
    elif w > h:
        offset = int((w-h)/2)
        return image[:, offset:w-offset, :]
    else:
        return image

img_names = os.listdir(inputFolder)
random.shuffle(img_names)
for im_index, img_name in enumerate(img_names):
    img_path = os.path.join(inputFolder, img_name)
    export_path = os.path.join(outputFolder, "{}.jpg".format(im_index))
    image = cv2.imread(img_path)
    # --- main action --- #
    #image = crop_around(image, 1.0 / 6)
    image = crop_square(image)
    # --- #
    cv2.imwrite(export_path, image)