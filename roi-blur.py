# This code blur out boats

import cv2
import guicfg as cfg
import numpy as np
import os
import json
import glob

def remove_ext(fname):
    return os.path.splitext(fname)[0]

def illustrate_mask(im_file_name):

    class_dict = cfg.classes_dict

    inp_json=remove_ext(im_file_name) + '.json'
    if not os.path.exists(inp_json):
        return
    else:
        fid = open(inp_json, 'r')
        str1 = fid.read()
        fid.close()
        y = json.loads(str1)
        objects_list = y['shapes']

    img = cv2.imread(im_file_name)
    img_ol = np.zeros_like(img)

    img_h, img_w = img.shape[0:2]
    anno_im = np.zeros_like(img)

    fill_colour = [(255,255,0)]

    for o in objects_list:
        pts = np.array(o['points']).astype(np.int32)
        if o['label'] == 'discard':
            classid = 0
        else:
            classid = class_dict[o['label']]

        if classid==0:
            fill_colour = (0,0,0)
        elif classid==1:
            fill_colour = (255,255,255)
        elif classid==2:
            fill_colour = (0,0,0)
        elif classid==3:
            fill_colour = (0,0,0)
        elif classid==4:
            fill_colour = (0,0,0)
        cv2.fillPoly(img_ol, [pts], fill_colour)

    return img_ol

files = glob.glob('/home/on99/Pictures/Pierwithsmallcracks_191205/*.JPG')
for fname in files:
    print(fname)
    img=cv2.imread(fname)
    clear=img.copy()
    blur_img=cv2.blur(img, (10,10))
    mask=illustrate_mask(fname)
    blur_img=np.bitwise_and(blur_img,mask)
    inv_mask=np.invert(mask)
    clear=np.bitwise_and(clear, inv_mask)
    clear = clear + blur_img
    cv2.imwrite(remove_ext(fname) + '.jpg', clear, [cv2.IMWRITE_JPEG_QUALITY, 98])
