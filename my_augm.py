import geotiff
import os
import cv2
import random
import numpy as np
from scipy import ndimage, misc

fname = R"C:\Users\dva\Pictures\20180313SA1_B05_6NE11B_W_16384_H_12288_X_0_2048_Y_2048_4096.tif"
im = geotiff.imread(fname)
#im = cv2.imread(fname, 1)

im = im.astype(np.float32)
im = im[:,:,0:6]

im = ndimage.rotate(im, -90, reshape=False)

im = im[:,:,0:3]

geotiff.imwrite(os.path.splitext(fname)[0]+'_augm.tif',im)

for n in range(10):
    random_bit = random.getrandbits(1)
    random_boolean = bool(random_bit)
    print(random_boolean)
