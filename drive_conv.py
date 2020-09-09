# This code convert index gif mask file to png UNET mask file
import PIL
import cv2
from PIL import Image
import numpy as np
import os
import glob

def remove_ext(inp):
    return os.path.splitext(inp)[0]

#files = glob.glob('*.gif')
files = ['/home/ins/Pictures/pier_small_set/DJI_0352s.png']
for fname in files:
    pil_img = Image.open(fname)
    img=np.array(pil_img)
    img = img//255
    cv2.imwrite(remove_ext(fname)+'.png', img)