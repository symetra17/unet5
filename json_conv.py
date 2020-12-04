import json
import cv2
import numpy as np
import os
import skimage
import skimage.transform
import guicfg as cfg
import glob
from importlib import reload
from collections import OrderedDict
import geotiff
import multiprocessing as mp
import random
import time
from itertools import repeat
from math import sin, cos, radians


def remove_ext(inp):
    # Remove filename extenstion, xxx123.jpg to xxx123
    return os.path.splitext(inp)[0]


def path_insert_folde(filename, folder):
    splited = os.path.split(filename)
    return os.path.join(splited[0], folder, splited[1])

def get_rand_angle():
    angle = random.randint(cfg.augm_angle_range[0], cfg.augm_angle_range[1])
    return angle

def read_img_anno(inplist, im_file_name, cls_name):
    
    img = geotiff.imread(im_file_name)  # 5-channel array included NIR and DSM.    
    img = np.nan_to_num(img)
    anno_im = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    cls_sub_list = cfg.get(cls_name).cls_sub_list

    for o in inplist:
        pts = np.array(o['points']).astype(np.int32)
        try:
            classid = cls_sub_list[o['label']]
        except:
            classid = 0
            if o['label'] != 'discard':
                print('Unexpected class label: ', o['label'])
        cv2.fillPoly(anno_im, [pts], (classid))
    
    for o in inplist:
        pts = np.array(o['points']).astype(np.int32)
        if o['label'] == 'discard':
            cv2.fillPoly(anno_im, [pts], (255))

    return img, anno_im


def skimage_rotate(im_file_name, img, n, angle, dn_scale):
    img_ls = []
    n_layer = img.shape[2]
    for k in range(n_layer):
        warp_dst = img[:,:,k]
        h = warp_dst.shape[0] 
        w = warp_dst.shape[1] 
        new_h = abs(h*cos(radians(angle))) + abs(w*sin(radians(angle)))
        new_w = abs(h*sin(radians(angle))) + abs(w*cos(radians(angle)))
        dx = new_w - w
        dy = new_h - h
        center = ( warp_dst.shape[1]//2, warp_dst.shape[0]//2 )
        rot_mat = cv2.getRotationMatrix2D( center, angle, 1.0 )
        rot_mat[0, 2] += int(dx/2)    # shift x
        rot_mat[1, 2] += int(dy/2)    # shift y
        img1 = cv2.warpAffine(warp_dst, rot_mat, 
            (int(new_w), int(new_h)), borderValue = -1)

        #img1 = skimage.transform.rotate(img[:,:,k], angle, resize=True, 
        #                mode='constant', cval=-1.0, preserve_range=True)

        if dn_scale != 1:
            img1 = cv2.resize(img1, None, fx=1/dn_scale, fy=1/dn_scale, 
                interpolation=cv2.INTER_AREA)
        img_ls.append(img1)

    h = img_ls[0].shape[0]
    w = img_ls[0].shape[1]

    out_img = np.zeros( (h,w,img.shape[2]), dtype=img_ls[0].dtype )
    for k, layer in enumerate(img_ls):
        out_img[:,:,k] = layer.copy()

    img1 = out_img.astype(np.half)
    np.save(os.path.splitext(im_file_name)[0] + '_im_rot_%d'%n, img1)     # save it for use next time
    # geotiff.imwrite('ROT_%d.tif'%n, img1[:,:,0:3].astype(np.uint8))

def skimage_rotate_annot(im_file_name, anno_im, n, angle, dn_scale):

    warp_dst = anno_im

    h = warp_dst.shape[0] 
    w = warp_dst.shape[1] 
    new_h = abs(h*cos(radians(angle))) + abs(w*sin(radians(angle)))
    new_w = abs(h*sin(radians(angle))) + abs(w*cos(radians(angle)))
    dx = new_w - w
    dy = new_h - h
    center = ( warp_dst.shape[1]//2, warp_dst.shape[0]//2 )
    rot_mat = cv2.getRotationMatrix2D( center, angle, 1.0 )
    rot_mat[0, 2] += int(dx/2)    # shift x
    rot_mat[1, 2] += int(dy/2)    # shift y
    anno_im1 = cv2.warpAffine(warp_dst, rot_mat, 
                (int(new_w), int(new_h)), borderValue=0)

    #anno_im1 = skimage.transform.rotate(anno_im, angle, resize=True, 
    #        mode='constant', cval=0, preserve_range=True)

    if dn_scale != 1:
        anno_im1 = cv2.resize(anno_im1, None, fx=1/dn_scale, fy=1/dn_scale, 
            interpolation=cv2.INTER_AREA)
    anno_im1 = anno_im1.astype(np.half)
    np.save(os.path.splitext(im_file_name)[0] + '_anno_rot_%d'%n, anno_im1)


def filter_save_training_patch(img, anno_im, ystart, yend, xstart,xend, 
        my_size, 
        outname, a_or_b, cls_cfg_discard_empty):

    sub_im = img[ystart:yend, xstart:xend, :]
    sub_im = sub_im.astype(np.float32)

    sub_anno = anno_im[ystart:yend, xstart:xend]

    if sub_anno.max() == 255:   # if any pixel is marked as discard in json file
        return

    # Pad image to 1024 divisible
    sub_img_h, sub_img_w = sub_im.shape[0:2]
    if sub_img_h < my_size:
        pady = my_size - sub_img_h
    else:
        pady=0
    if sub_img_w < my_size:
        padx = my_size - sub_img_w
    else:
        padx=0
    
    sub_im = np.pad(sub_im, ((0,pady), (0,padx), (0,0) ) , 
                    mode='constant',constant_values=-1.0)
    sub_anno = np.pad(sub_anno, ((0,pady), (0,padx)))

    assert sub_im.shape[0] == my_size
    assert sub_im.shape[1] == my_size
    assert sub_anno.shape[0] == my_size
    assert sub_anno.shape[1] == my_size

    # Some images contain large blank black or white area, and these 
    # area should not be included in training.

    # too many pixel has rotated out of boundary area
    idx = np.where(sub_im[:,:,0] < 0)
    empty_ratio = len(idx[0]) / (my_size*my_size)
    if empty_ratio > 0.5:
        return

    if int(np.percentile(sub_anno, 99)) == 0:
        v = random.random()
        if v < cls_cfg_discard_empty:  # keep some of background picture
            return                   # discard this picture
    
    sub_im_0 = sub_im[:,:,0].copy()
    sub_im_0[sub_im_0<0] = 128.0
    sub_im[:,:,0] = sub_im_0


    if sub_im.shape[2] > 1:
        sub_im_1 = sub_im[:,:,1].copy()
        sub_im_1[sub_im_1<0] = 128.0
        sub_im[:,:,1] = sub_im_1

    if sub_im.shape[2] > 2:
        sub_im_2 = sub_im[:,:,2].copy()
        sub_im_2[sub_im_2<0] = 128.0
        sub_im[:,:,2] = sub_im_2

    if sub_im.shape[2] > 3:
        sub_im_3 = sub_im[:,:,3].copy()
        sub_im_3[sub_im_3<0] = 128.0
        sub_im[:,:,3] = sub_im_3

    if sub_im.shape[2] > 4:
        sub_im_4 = sub_im[:,:,4].copy()
        sub_im_4[sub_im_4<0] = 0
        sub_im[:,:,4] = sub_im_4

    outpath = path_insert_folde(outname, 'slice'+a_or_b)
    outpath = path_insert_folde(outpath, 'annotation')
    cv2.imwrite(outpath + '.png', sub_anno.astype(np.float32))
    outpath = path_insert_folde(outname + '.tif', 'slice'+a_or_b)
    outpath = path_insert_folde(outpath, 'image')
    geotiff.imwrite(outpath, sub_im.astype(np.float32))

    # RGB jpg for manual inspection
    geotiff.imwrite(os.path.splitext(outpath)[0]+'.jpg', 
            sub_im[:,:,0:3].astype(np.uint8))


def split_label(inplist, im_file_name, cls_name, a_or_b):
    
    reload(cfg)
    cls_cfg = cfg.get(cls_name)
    try:
        down_scale = cls_cfg.down_scale
    except:
        down_scale = 1

    my_size = cls_cfg.my_size * down_scale
    random.seed()
        
    if not cfg.augm_rotation:

        img, anno_im = read_img_anno(inplist, im_file_name, cls_name)
        img = cv2.resize(img, None, fx=1/down_scale, fy=1/down_scale, 
                interpolation=cv2.INTER_AREA)
        anno_im = cv2.resize(anno_im, None, fx=1/down_scale, fy=1/down_scale, 
                interpolation=0)
    else:

        anglels = [-30, -20, -10, 0, 10, 20, 30]
        angle_idx = random.randint(0, len(anglels)-1)
        fname_im = os.path.splitext(im_file_name)[0] + '_im_rot_%d'%angle_idx
        fname_anno = os.path.splitext(im_file_name)[0] + '_anno_rot_%d'%angle_idx
        if os.path.exists(fname_im + '.npy') and os.path.exists(fname_anno + '.npy'):
            pass
        else:
            t0 = time.time()
            print('Generating rotated img: ', os.path.split(im_file_name)[-1])
            img, anno_im = read_img_anno(inplist, im_file_name, cls_name)

            for n, angle in enumerate(anglels):
                skimage_rotate(im_file_name, img, n, angle, down_scale)

            for n, angle in enumerate(anglels):
                skimage_rotate_annot(im_file_name, anno_im, n, angle, down_scale)

            t1 = time.time()
            print('rotation augm time:', int(t1-t0), 'sec')

        img     = np.load(fname_im + '.npy')
        anno_im = np.load(fname_anno + '.npy').astype(np.uint8)

    assert img.shape[0:2] == anno_im.shape[0:2]

    my_size = cls_cfg.my_size
    augm_x = int(cfg.augm_translate) * random.randint(0, my_size-1)
    augm_y = int(cfg.augm_translate) * random.randint(0, my_size-1)
    
    for n in range(1+int(img.shape[1]/my_size)):
        for m in range(1+int(img.shape[0]/my_size)):
    
            xstart = n * my_size + augm_x
            xend   = xstart + my_size
            if xstart >= img.shape[1]:
                continue
            if xend > img.shape[1]:
                xend = img.shape[1]

            ystart = m * my_size + augm_y
            yend   = ystart + my_size
            if ystart >= img.shape[0]:
                continue
            if yend > img.shape[0]:
                yend = img.shape[0]
            
            outname = remove_ext(im_file_name) + '_W_%d_H_%d_X_%d_%d_Y_%d_%d'%(
                img.shape[1],img.shape[0],xstart,xend,ystart,yend)

            filter_save_training_patch(img, anno_im, ystart, yend, xstart,xend, 
                my_size, outname, a_or_b, cls_cfg.discard_empty)

            

def mp_split(fname, cls_name, a_or_b):

    inp_json = remove_ext(fname) + '.json'
    if not os.path.exists(inp_json):
        print('Error: Could not find label json file for ', fname)
        return

    fid = open(inp_json, 'r')
    str1 = fid.read()
    fid.close()
    y = json.loads(str1)
    nobj = len(y['shapes'])
    objects_list = y['shapes']
    split_label(objects_list, fname, cls_name, a_or_b)

def recut(foder, cls_name, a_or_b):

    def clean_folder(folder):
        try:
            os.mkdir(folder)
        except:
            pass
        to_del = glob.glob(os.path.join(folder,'*.*'))
        for f in to_del:
            os.remove(f)
    
    def create_clean_folder(folder):
        try:
            os.mkdir(folder)
        except:
            pass
        clean_folder(folder)

    if os.name == 'nt':
        ftypes = ['jpg', 'jpeg', 'tif', 'png', 'bmp']
    else:
        ftypes = ['jpg', 'JPG', 'jpeg', 'JPEG', 'tif', 'TIF', 'png', 'PNG', 'bmp', 'BMP']
    files = []
    for ft in ftypes:
        files.extend(glob.glob(os.path.join(foder, '*.'+ft)))
    if len(files)==0:
        msg = 'No image input files found, training could not start'
        print(msg)
        return
    im_path = os.path.join(foder, 'slice' + a_or_b, 'image')
    ann_path = os.path.join(foder, 'slice' + a_or_b, 'annotation')
    create_clean_folder(os.path.join(foder, 'slice' + a_or_b))
    create_clean_folder(ann_path)
    create_clean_folder(im_path)

    pool = mp.Pool(processes=8)
    for n in range(2):
        args = zip( files, repeat(cls_name), repeat(a_or_b) )
        pool.starmap(mp_split, args)
    pool.terminate()


def illustrate_dsm_mask(im_file_name, class_dict):
    inp_json=remove_ext(im_file_name) + '.json'
    if not os.path.exists(inp_json):
        return
    else:
        fid = open(inp_json, 'r')
        str1 = fid.read()
        fid.close()
        y = json.loads(str1)
        objects_list = y['shapes']

    img = geotiff.imread(im_file_name)
    img_dsm = img[:,:,4].copy()
    img_dsm = cv2.cvtColor(img_dsm, cv2.COLOR_GRAY2BGR)
    img_dsm_ol = img_dsm.copy()
    fill_colour = [(255,255,0)]
    for o in objects_list:
        pts = np.array(o['points']).astype(np.int32)
        classid = 1   #class_dict[o['label']]
        if classid==0:
            fill_colour = (0,0,0)
        elif classid==1:
            fill_colour = (0,0,255)
        elif classid==2:
            fill_colour = (0,255,0)
        elif classid==3:
            fill_colour = (255,0,0)
        elif classid==4:
            fill_colour = (255,255,0)
        cv2.fillPoly(img_dsm_ol, [pts], (0,0,255) )
        img_dsm_ol = cv2.polylines( img_dsm_ol, [pts], True, (255,0,0), 10 )

    anno_im_dsm = img_dsm * 0.7 + img_dsm_ol * 0.3
    geotiff.imwrite(remove_ext(im_file_name)+'_label_dsm.jpg', anno_im_dsm.astype(np.uint8))


def illustrate_mask(im_file_name, json_fname):

    fid = open(json_fname, 'r')
    str1 = fid.read()
    fid.close()
    y = json.loads(str1)
    objects_list = y['shapes']

    img = geotiff.imread(im_file_name)
    img = img[:,:,0:3]
    img_ol = img.copy()
    img_ol = img_ol.astype(np.uint8)
    img_h, img_w = img.shape[0:2]
    label_only_im = np.zeros_like(img)
    fill_colour = [(255,255,0)]
    for o in objects_list:
        pts = np.array(o['points']).astype(np.int32) 
        if o['label'] == 'discard':
            cv2.fillPoly(img_ol, [pts], (128,128,128) )
        else:
            classid = 1
            if classid==0:
                fill_colour = (0,0,0)
            elif classid==1:
                fill_colour = (0,255,255)   # (R,G,B)   
            img_ol = cv2.polylines( img_ol, [pts], True, (0,255,0), 4 )
        
    anno_im = img * 0.3 + img_ol * 0.7
    geotiff.imwrite(remove_ext(im_file_name)+'_label.jpg', anno_im.astype(np.uint8))

def illustrate_mask_bw(json_fname):

    fid = open(json_fname, 'r')
    str1 = fid.read()
    fid.close()
    y = json.loads(str1)
    objects_list = y['shapes']
    imageHeight = int(y['imageHeight'])
    imageWidth = int(y['imageWidth'])

    img_ol = np.zeros((imageHeight, imageWidth), dtype=np.uint8)
    for o in objects_list:
        pts = np.array(o['points']).astype(np.int32)
        cv2.fillPoly(img_ol, [pts], (255) )

    for o in objects_list:
        pts = np.array(o['points']).astype(np.int32)
        if o['label'] == 'discard':
            cv2.fillPoly(img_ol, [pts], (128) )
            
    cv2.imwrite('img_ol.png',img_ol)
    return img_ol

def get_class_name(img_files):
    nobj = 0
    cls_dict = OrderedDict()
    for fname in img_files:
        inp_json = remove_ext(fname) + '.json'
        if not os.path.exists(inp_json):
            pass
        else:
            fid = open(inp_json, 'r')
            str1 = fid.read()
            fid.close()
            y = json.loads(str1)
            polygon_objs = y['shapes']
            nobj += len(polygon_objs)
            for obj in polygon_objs:
                cls_name = obj['label']
                cls_dict[cls_name] = True
    return list(cls_dict.keys())


def calculate_accuracy(fname_json, fname_pred, out_txt_name):

    fid = open(out_txt_name, 'a')
    
    gt = illustrate_mask_bw(fname_json)
    pred = cv2.imread(fname_pred, 2)
    mask = 255 * np.ones_like(pred)
    idx = np.where(gt==128)
    mask[idx] = 0    
    assert pred.shape == gt.shape
    TP = np.bitwise_and(pred, gt)
    TP = np.bitwise_and(TP, mask)
    nTP = np.count_nonzero(TP)

    FP = np.bitwise_and( pred, np.bitwise_not(gt) )
    FP = np.bitwise_and(FP, mask)

    nFP = np.count_nonzero(FP)

    gt = np.bitwise_and(gt, mask)
    nGT = np.count_nonzero(gt)

    print(os.path.split(fname_json)[-1])
    fid.write(os.path.split(fname_json)[-1])
    fid.write('\n')

    # How many selected items are relevant
    precis = 100*nTP/(nTP + nFP)
    print('Precision: %.1f%%'%precis)
    fid.write('Precision: %.1f%%\n'%precis)

    # How many relevant items are selected
    recall = 100*nTP/nGT
    print('Recall   : %.1f%%'%recall)
    fid.write('Recall   : %.1f%%\n'%recall)
    fid.write('\n')
    fid.close()

    return precis, recall


def illustrate_mask_dir(img_dir, json_dir):

    img_files = glob.glob(os.path.join(img_dir,'*result.tif'))
    json_files = glob.glob(os.path.join(json_dir,'*.json'))
    assert len(img_files) == len(json_files)
    pair_ls = zip(img_files, json_files)
    pool = mp.Pool(processes=12)
    pool.starmap(illustrate_mask, pair_ls)
    pool.terminate()


def calculate_accuracy_dir(json_dir, pred_dir):
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    pred_files = glob.glob(os.path.join(pred_dir, '*result_bw.bmp'))
    out_files = []
    out_txt_name = os.path.join(os.path.dirname(pred_files[0]), 'accuracy.txt')
    for n in range(len(json_files)):
        calculate_accuracy(json_files[n], pred_files[n], out_txt_name)

if __name__=='__main__':

    #illustrate_mask_dir(R"C:\Users\dva\unet5\result", R"C:\Users\dva\unet_dsm\test-set\gt" )
    calculate_accuracy_dir(R"C:\Users\dva\unet_dsm\test-set\gt", R"C:\Users\dva\unet5\result")

    quit()

    inp_dir = R'C:\Users\dva\unet_dsm\weights\Squatter\TS-1'
    t0 = time.time()
    xxx(inp_dir, 'Squatter', 'a')
    t1 = time.time()
    print('Total time:  %.0f'%(t1-t0))


