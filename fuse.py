import cv2
import numpy as np
import glob
import os

def remove_ext(fname):
    return os.path.splitext(fname)[0]


files = glob.glob('/home/on99/Pictures/Pierwithsmallcracks_191205/*.JPG')

for fname in files:

    img=cv2.imread(fname)
    fname_mask = remove_ext(os.path.split(fname)[-1])+'_result.bmp'

    mask1 = cv2.imread('/home/on99/Pictures/Pierwithsmallcracks_191205/willfuse/boatbw/'+fname_mask)
    mask2 = cv2.imread('/home/on99/Pictures/Pierwithsmallcracks_191205/willfuse/crack_only_result/'+fname_mask)

    inv_mask = np.invert(mask1)
    out = np.bitwise_and(mask2, inv_mask)
    
    mask3 = out
    mask3[:,:,0]=0
    mask3[:,:,1]=0
    #img = img//2
    img = img//2 + np.bitwise_or(img, mask3)//2
    
    cv2.imwrite(remove_ext(fname_mask) + '_out.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 97])

