import numpy as np
import cv2
from glob import glob
import os

def remove_ext(fname):
    return os.path.splitext(fname)[0]

files = glob('/home/ins/Pictures/Defect photos/slice/annotation/*.png')

for n, fname in enumerate(files):
    img = cv2.imread(fname)
    img = img * 50
    cv2.imwrite(remove_ext(fname) + '_view.png', img)
    print(n)
