import numpy as np
import cv2
import glob
import os
import random

def get_statistics(path):
    blank=1
    non_blank=0
    files = glob.glob(os.path.join(path, 'annotation', '*.png'))
    for f in files:
        img = cv2.imread(f)
        mv = img.max()
        if mv == 0:
            blank += 1
        else:
            non_blank += 1
    print('blank', blank, '  non blank', non_blank)
    return non_blank/blank
