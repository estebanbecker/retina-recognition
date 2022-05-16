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
    
    
    floodfill_mask = cv2.bitwise_not(src=im_floodfill)
    floodfill_mask = cv2.threshold(floodfill_mask, 1, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('dil_kernel', floodfill_mask)
    floodfill_mask = cv2.erode(floodfill_mask, dil_kernel, iterations=1)
    cv2.imshow("im_floodfill", floodfill_mask)
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    im_contrast = clahe.apply(im_gray)

    im_contrast = 255-im_contrast
    im_contrast = clahe.apply(im_contrast)

    im_gauss = cv2.GaussianBlur(im_contrast, (7, 7),0)

    im_threshold= cv2.adaptiveThreshold(im_gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11, C=0)
    cv2.imshow('threshold', im_threshold)
    img=cv2.bitwise_and(im_threshold, floodfill_mask)
    cv2.imshow("winname", img)
    cv2.waitKey(0)
    img=cv2.GaussianBlur(img, (5, 5),0)
    img=cv2.bitwise_not(img)
    img=cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,1)))
    img=cv2.bitwise_not(src=img)
    cv2.imshow("winname", img)
    cv2.waitKey()  

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    
    #remove small components (area < 500)
    img2 = np.zeros((output.shape), dtype=np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= 500:
            img2[output == i + 1] = 255

    cv2.imshow("winname", img2)
    cv2.waitKey()    

    skel_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    skel = np.zeros(img_obj_filtered.shape, np.uint8)
    while True:
        opened = cv2.morphologyEx(img_obj_filtered, cv2.MORPH_OPEN, skel_element)
        temp = cv2.subtract(img_obj_filtered, opened)
        eroded = cv2.erode(img_obj_filtered, skel_element)
        skel = cv2.bitwise_or(skel, temp)
        img_obj_filtered = eroded.copy()
        if cv2.countNonZero(img_obj_filtered)==0:
            break

    cv2.imshow('img', img)
    cv2.waitKey(0)
    labels.append(filename[7])
