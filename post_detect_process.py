import cv2 as cv
import json
import numpy as np

fname = R'/home/on99/Pictures/Buildings/20200707/20190125SA1_A06_11SW14B_Focal_small_result_bw.bmp'
img = cv.imread(fname,2)

img_dsm = cv.imread(R'/home/on99/Pictures/Buildings/20200707/20190125SA1_A06_11SW14B_Focal_small.jpg',2)

dil_type = cv.MORPH_ELLIPSE
dil_size = 10
elem = cv.getStructuringElement(dil_type, (2*dil_size + 1, 2*dil_size+1), 
        (dil_size, dil_size))
img = cv.dilate(img, elem)

contours, hierarchy = cv.findContours(img, cv.RETR_TREE, 
        cv.CHAIN_APPROX_NONE)

img2 = np.zeros((img.shape[0], img.shape[1], 3),dtype=np.uint8)
print(img2.shape)

for n, ctr in enumerate(contours):
    ctr = np.squeeze(ctr, 1)
    val = []
    for n in range(ctr.shape[0]):
        x = ctr[n,0]
        y = ctr[n,1]
        img2[y, x, :] = (0,255,255)        
        val.append(img_dsm[y,x])
    val = np.array(val)    
    print('cont ', n, ' min point', val.min())

#cv.drawContours(img2, contours, 10, (100,100,100), thickness=3)
cv.imwrite('out.bmp',img2)
