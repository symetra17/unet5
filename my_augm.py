import geotiff
import os
import cv2
import random
import numpy as np
from scipy import ndimage, misc
import imgaug as ia
from imgaug import augmenters as iaa
import json
import skimage

fname = R"C:\Users\dva\unet5\weights\Squatter\20180313SA1_B05_6NW13D.tif"
im = geotiff.imread(fname)
im = im.astype(np.uint8)
print(im.dtype)

im = skimage.transform.rotate(im, 10, resize=True, mode='constant', cval=125, preserve_range=True)

print(im.dtype)
geotiff.imwrite(os.path.splitext(fname)[0]+'_augm.tif',im)
