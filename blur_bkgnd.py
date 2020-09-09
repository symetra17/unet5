from glob import glob
import cv2
import json
import numpy as np
import os
import time
from cv2 import imread

fea_siz = 20   # feather size

def do_thing(fname, outfname):
    fname_json = os.path.splitext(fname)[0] +'.json'
    fid = open(fname_json, 'r')
    str1 = fid.read()
    fid.close()
    y = json.loads(str1)
    nobj = len(y['shapes'])
    objects_list = y['shapes']
    img = cv2.imread(fname)
    img_h, img_w = img.shape[0:2]
    mask = np.zeros_like(img)
    #mask_inv = np.zeros_like(img)
    for o in objects_list:
        pts = np.array(o['points']).astype(np.int32)
        cv2.fillPoly(mask, [pts], (1,1,1))
        #cv2.fillPoly(mask_inv, [pts], (255,255,255))
    alpha = cv2.blur(mask.astype(np.float), (fea_siz,fea_siz))
    img_b = cv2.GaussianBlur(img,(51,51),0)
    img_a = img.astype(np.float)
    img_b = img_b.astype(np.float)
    img_blend = img_a*alpha + img_b*(1-alpha)
    cv2.imwrite(outfname, img_blend, [cv2.IMWRITE_JPEG_QUALITY, 98])

def blur_bkgnd(fname, fname2, outname_noext):
    img = imread(fname)
    mask = imread(fname2)
    mask = mask.astype(np.float)/255
    alpha = cv2.blur(mask, (fea_siz,fea_siz))
    img_a = img.astype(np.float)
    img_b = cv2.GaussianBlur(img,(51,51),0)
    img_blend = img_a*alpha + img_b*(1-alpha)
    cv2.imwrite(outname_noext + '.jpg', img_blend, [cv2.IMWRITE_JPEG_QUALITY, 98])


if __name__=='__main__':
    fname = r'/media/on99/7eaa8f84-b1ed-41ce-8ecc-4a1740d2719a/home/insight/asd_subset_concrete/DJI_0521.JPG'
    fname2 = r'/media/on99/7eaa8f84-b1ed-41ce-8ecc-4a1740d2719a/home/insight/asd_subset_concrete/DJI_0521_result.jpg'
    blur_bkgnd(fname, fname2, 'my_output')
    quit()

    dir = r'E:\asd_2000m\asd_subset_concrete\*.jpg'
    files = glob(dir)
    t0=time.time()
    for f in files:
        print(f)
        outname = os.path.join(r'E:\asd_2000m\asd_subset_crack', os.path.split(f)[-1])
        do_thing(f,outname)
    print(int(time.time()-t0))
