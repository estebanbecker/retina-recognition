from genericpath import isfile
from ntpath import join
from os import listdir
import cv2
import numpy as np


filename_list = [f for f in listdir('RIDB') if isfile(join('RIDB/',f))]

images = []
labels = []

for filename in filename_list:
    img = cv2.imread('RIDB/'+filename)

    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_floodfill = im_gray.copy()
    
    cv2.floodFill(im_floodfill, None,(0,0),255)

    h, w = im_floodfill.shape[:2]

    floodfill_mask = np.zeros((h+2, w+2), np.uint8)

    dil_kernel = np.ones((13, 13), np.uint8)

    im_bitwise_not = cv2.bitwise_not(im_floodfill)
    im_binarization = cv2.adaptiveThreshold(im_bitwise_not,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,2)
    cv2.imshow("g",im_binarization)
    cv2.waitKey()

    labels.append(filename[7])
