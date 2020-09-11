import geotiff
import os
import cv2
import random
import numpy as np

fname = R"C:\Users\echo\Code\unet5\weights\Farmland\20180103SA1_B05_2SE22B (Custom).TIF"
im = geotiff.imread(fname)
im = im.astype(np.float32)

im = cv2.transpose(im)

im = im[:,:,0:3]

geotiff.imwrite(os.path.splitext(fname)[0]+'_augm.tif',im)

for n in range(10):
    random_bit = random.getrandbits(1)
    random_boolean = bool(random_bit)
    print(random_boolean)
