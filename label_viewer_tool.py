import os
import cv2
import numpy as np
import time
import json
import geotiff
import guicfg as cfg

inp        = R"C:\Users\echo\Desktop\New folder\20180313SA1_B05_6NW20B.tif"
inp_json   = R"C:\Users\echo\Desktop\New folder\20180313SA1_B05_6NW20B.json"
outp       = R"C:\Users\echo\Desktop\New folder\20180313SA1_B05_6NW20B-dn2.tif"
outp_label = R"C:\Users\echo\Desktop\New folder\20180313SA1_B05_6NW20B-dn2-label.tif"

def draw_label(im_file_name, cls_name):

    fid = open(inp_json, 'r')
    str1 = fid.read()
    fid.close()
    y = json.loads(str1)
    objects_list = y['shapes']

    img = geotiff.imread(im_file_name)
    img = img.astype(np.uint8)

    anno_im = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    ntotal_out = 0
    cls_sub_list = cfg.get(cls_name).cls_sub_list
    for o in objects_list:
        pts = np.array(o['points']).astype(np.int32) // 2
        try:
            classid = cls_sub_list[o['label']]
        except:
            classid = 0
            print('Unexpected class label: ', o['label'])
        cv2.fillPoly(anno_im, [pts], (classid))
        print(img.shape, img.dtype)

        #cv2.fillPoly(img[:,:,0], [pts], (255))
        cv2.drawContours(img[:,:,0], [pts], -1, (255), 3)

        ntotal_out += 1

    geotiff.imwrite(outp_label, img)

im = geotiff.imread(inp)
im = im[:,:,0:3]
im = cv2.resize(im, None, fx=0.5, fy=0.5)
im = im.astype(np.uint8)
print('write')
geotiff.imwrite(outp, im)

draw_label(outp, 'Squatter')


