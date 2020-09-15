import json
import cv2
import numpy as np
import os
import skimage
import guicfg as cfg
import glob
from importlib import reload
from collections import OrderedDict
import geotiff
import multiprocessing as mp
import random

#PIL.Image.MAX_IMAGE_PIXELS = None

def remove_ext(inp):
    # Remove filename extenstion, xxx123.jpg to xxx123
    return os.path.splitext(inp)[0]


def path_insert_folde(filename, folder):
    splited = os.path.split(filename)
    return os.path.join(splited[0], folder, splited[1])


def split_label(inplist, im_file_name, cls_name, remove_blank=True):

    reload(cfg)
    cls_cfg = cfg.get(cls_name)
    try:
        down_scale = cls_cfg.down_scale
    except:
        down_scale = 1

    my_size = cls_cfg.my_size * down_scale
    random.seed()
    augmen_x = random.randint(0, my_size)
    augmen_y = random.randint(0, my_size)

    img = geotiff.imread(im_file_name)  # 5-channel array included NIR and DSM.

    print('padding')
    # Pad image to 1024 divisible
    img_h, img_w = img.shape[0:2]
    if img_h%my_size != 0:
        pady = my_size - (img_h%my_size)
    else:
        pady=0
    if img_w%my_size != 0:
        padx = my_size - (img_w%my_size)
    else:
        padx=0
    img = np.pad(img, ((0,pady), (0,padx), (0,0) ) , mode='mean')
    img_h, img_w = img.shape[0:2]
    #

    print('fill polygon')
    anno_im = np.zeros((img_h, img_w), dtype=np.uint8)
    ntotal_out = 0
    cls_sub_list = cfg.get(cls_name).cls_sub_list
    for o in inplist:
        pts = np.array(o['points']).astype(np.int32)
        try:
            classid = cls_sub_list[o['label']]
        except:
            classid = 0
            print('Unexpected class label: ', o['label'])

        cv2.fillPoly(anno_im, [pts], (classid))
        ntotal_out += 1

    ##Note: Possible to insert rotation in here
    angle = random.randint(-30, 30)
    img = skimage.transform.rotate(img, angle, resize=True, mode='constant', cval=125, preserve_range=True)
    anno_im = skimage.transform.rotate(anno_im, angle, resize=True, mode='constant', cval=125, preserve_range=True)

    for n in range(int(img.shape[1]/my_size)):
        for m in range(int(img.shape[0]/my_size)):
    
            xstart = n * my_size + augmen_x
            xend   = xstart + my_size
            if xend > img.shape[1]:
                continue

            ystart = m * my_size + augmen_y
            yend   = ystart + my_size
            if yend > img.shape[0]:
                continue
            
            outname=remove_ext(im_file_name) + '_W_%d_H_%d_X_%d_%d_Y_%d_%d.jpg'%(
                img.shape[1],img.shape[0],xstart,xend,ystart,yend)

            sub_im = img[ystart:yend, xstart:xend, :]
            sub_anno = anno_im[ystart:yend, xstart:xend]

            if remove_blank:
                # Some images contain large blank black or white area, and these 
                # area should not be included in training.
                mv = sub_im.mean()
                if mv==0 or mv==255:
                    continue
            if down_scale != 1:
                sub_im = cv2.resize(sub_im, None, fx=1/down_scale, fy=1/down_scale,
                        interpolation=0)
                sub_anno = cv2.resize(sub_anno, None, fx=1/down_scale, fy=1/down_scale,
                        interpolation=0)
            outpath = path_insert_folde(remove_ext(outname), 'slice')
            outpath = path_insert_folde(outpath, 'annotation')

            cv2.imwrite(outpath + '.png', sub_anno)
            outpath = path_insert_folde(remove_ext(outname) + '.tif', 'slice')
            outpath = path_insert_folde(outpath, 'image')
            geotiff.imwrite(outpath,sub_im)
            
    print('Total output count', ntotal_out)
    
def mp_split(files, cls_name):
    for fname in files:
        inp_json = remove_ext(fname) + '.json'
        if not os.path.exists(inp_json):
            pass
        else:
            fid = open(inp_json, 'r')
            str1 = fid.read()
            fid.close()
            y = json.loads(str1)              
            nobj = len(y['shapes'])
            objects_list = y['shapes']
            split_label(objects_list, fname, cls_name)
        return

def xxx(foder,cls_name):
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
    im_path = os.path.join(foder, 'slice', 'image')
    ann_path = os.path.join(foder, 'slice', 'annotation')
    create_clean_folder(os.path.join(foder, 'slice'))
    create_clean_folder(ann_path)
    create_clean_folder(im_path)

    nf = len(files)
    n_thread = mp.cpu_count() - 2
    if n_thread<=0:
        n_thread = 1
    if n_thread > nf:
        n_thread = nf
    if n_thread > 3:
        n_thread = 3

    file_groups = []
    for n in range((n_thread-1)):
        file_groups.append(files[n*nf//n_thread: (n+1)*nf//n_thread])
    file_groups.append(files[(n_thread-1)*nf//n_thread: ])

    proc_group = []
    for n in range(n_thread):
        proc_group.append(mp.Process(target=mp_split, args=(file_groups[n],cls_name,)))

    for n in range(n_thread):
        proc_group[n].start()

    for n in range(n_thread):
        proc_group[n].join()

def illustrate_mask(im_file_name, class_dict):

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
    img_ol = img.copy()

    img_h, img_w = img.shape[0:2]
    anno_im = np.zeros_like(img)
    label_only_im = np.zeros_like(img)

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
            fill_colour = (0,0,255)
        elif classid==2:
            fill_colour = (0,255,0)
        elif classid==3:
            fill_colour = (255,0,0)
        elif classid==4:
            fill_colour = (255,255,0)
        cv2.fillPoly(img_ol, [pts], fill_colour)
        cv2.fillPoly(label_only_im, [pts], (255,255,255))

    cv2.imwrite(remove_ext(im_file_name)+'_label_no_bkg.bmp', label_only_im)

    anno_im = img/2 + img_ol/2
    cv2.imwrite(remove_ext(im_file_name)+'_label.jpg', anno_im)

    
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

if __name__=='__main__':
    xxx(R"C:\Users\dva\unet5\weights\Squatter\TS", 'Squatter')
    quit()

    dir = '/mnt/crucial-ssd/home/insight/Pictures/lands_1963_house_ts'
    ftypes = ['*.JPG', '*.jpg', '*tif']
    files = []
    for ftype in ftypes:
        files.extend(glob.glob(dir+'/'+ftype))

    for f in files:
        print(f)
        illustrate_mask(f,cfg.classes_dict)
    quit(0)


    inp='/home/on99/Downloads/IMG_3588.jpg'
    out_dir=os.path.join(os.path.split(inp)[0] , 'slice')
    try:
        os.mkdir(out_dir)
    except:
        pass

    im_folder = os.path.join(out_dir,'train')
    ann_folder = os.path.join(out_dir,'annotation')
    try:
        os.mkdir(ann_folder)
    except:
        pass
    try:
        os.mkdir(im_folder)
    except:
        pass

