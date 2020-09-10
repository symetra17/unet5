import geotiff
import os
import cv2
import random

fname = R"C:\Users\echo\Code\unet5\weights\Farmland\20180103SA1_B05_2SE22B (Custom).TIF"
img = geotiff.imread(fname)
img = cv2.flip(img, 1)
img = img[:,:,0:3]
geotiff.imwrite(os.path.splitext(fname)[0]+'_augm.tif',img)

for n in range(10):
    random_bit = random.getrandbits(1)
    random_boolean = bool(random_bit)
    print(random_boolean)
