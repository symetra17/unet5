import numpy as np
import cv2
import glob
import os
import random

def delete_blank(base_dir, remove_portion):
    blank=0
    non_blank=0
    dir1 = os.path.join(base_dir, R'annotation', R'*.png')
    files = glob.glob(dir1)
    for f in files:
        img = cv2.imread(f)
        mv = img.max()
        if mv == 0:
            if random.random() < remove_portion:
                fname = os.path.split(f)[-1]
                body = os.path.splitext(fname)[0]
                im_name = os.path.join(base_dir, R'image', body +'.jpg')
                if os.path.exists(im_name):
                    os.remove(f)
                    os.remove(im_name)
            else:
                blank += 1
        else:
            non_blank += 1
    print('blank', blank, '  non blank', non_blank)
    return non_blank/blank

if __name__=='__main__':
    delete_blank(R'C:\Users\dva\Pictures\PVPanel(B05_6)\slice', 0.1)
