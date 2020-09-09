import cv2
import numpy as np
import glob
import json_conv
import guicfg as cfg
import os

def getAccuracy(fname):
    json_conv.illustrate_mask(fname, cfg.classes_dict)
    f = os.path.splitext(fname)[0] + R'_label_no_bkg.bmp'
    img1 = cv2.imread(f)
    f = os.path.splitext(fname)[0] + R'_result_bw.bmp'
    img2 = cv2.imread(f)
    if img2 is None:
        raise Exception('Could not read ' + f)
    x = np.logical_xor(img1, img2)
    y = x*255
    npix_loss = np.count_nonzero(y)
    ntotalpix = img2.shape[0]*img2.shape[1]
    accuracy = 100*npix_loss/ntotalpix
    fname = os.path.splitext(fname)[0] + '_xor.bmp'
    cv2.imwrite(fname, y)    
    return accuracy

if __name__ == '__main__':
    fname = '/mnt/crucial-ssd/home/insight/Pictures/lands_1963_house_ts/stash/63ST7SW15c2013302.tif'
    accuracy = getAccuracy(fname)
    print('Percentage loss %.2f'%accuracy)

