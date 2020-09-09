import cv2
import PIL
import glob
from PIL import Image, ImageSequence
import numpy as np
import os
import guicfg as cfg

def remove_ext(inp):
    return os.path.splitext(inp)[0]

def path_insert_folde(filename, folder):
    splited = os.path.split(filename)
    return os.path.join(splited[0], folder, splited[1])

def multilayer_tif_2_single_layer_npy(inp_fname):
    pim = PIL.Image.open(inp_fname)
    label_np=np.zeros((pim.height, pim.width, 3), dtype=np.uint8)
    for i, page in enumerate(ImageSequence.Iterator(pim)):
        if i == 0:
            img_npy = np.array(page)
        else:
            pix = np.array(page).astype(np.bool)
            pix = pix.astype(np.uint8)*i
            label_np = np.bitwise_or(label_np, pix[:,:,0:3])
    #outpath = remove_ext(inp_fname) + '.png'
    #cv2.imwrite(outpath, label_np)
    return img_npy, label_np

def split_image(img, im_file_name):
    print('---', im_file_name)
    my_size = cfg.my_size
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
    for n in range(int(img_w/my_size)):
        for m in range(int(img_h/my_size)):
            xstart = (n+0)*my_size
            xend   = (n+1)*my_size
            ystart = (m+0)*my_size
            yend   = (m+1)*my_size            
            outname=remove_ext(im_file_name) + '_W_%d_H_%d_X_%d_%d_Y_%d_%d.jpg'%(
                img_w,img_h,xstart,xend,ystart,yend)
            sub_im = img[ystart:yend, xstart:xend, :]
            outpath = path_insert_folde(remove_ext(outname) + '.jpg', 'slice')
            outpath = path_insert_folde(outpath, 'image')
            sub_im = cv2.cvtColor(sub_im, cv2.COLOR_BGR2RGB)
            cv2.imwrite(outpath, sub_im, [cv2.IMWRITE_JPEG_QUALITY, 98])
            print(outpath)
            

def split_annotation(img, im_file_name):
    my_size = cfg.my_size
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
    nn=0
    for n in range(int(img_w/my_size)):
        for m in range(int(img_h/my_size)):    
            xstart = (n+0)*my_size
            xend   = (n+1)*my_size
            ystart = (m+0)*my_size
            yend   = (m+1)*my_size
            outname=remove_ext(im_file_name) + '_W_%d_H_%d_X_%d_%d_Y_%d_%d.jpg'%(
                img_w,img_h,xstart,xend,ystart,yend)
            frame = img[ystart:yend, xstart:xend, :]
            outpath = path_insert_folde(remove_ext(outname), 'slice')
            outpath = path_insert_folde(outpath, 'annotation')
            cv2.imwrite(outpath + '.png', frame)

def process_tif_dir(inp):
    # input: folder that contains multilayer tif file
    print(os.path.join(inp,'slice'))
    try:
        os.mkdir(os.path.join(inp,'slice'))
    except:
        pass
    try:
        os.mkdir(os.path.join(inp,'slice','image'))
    except:
        pass
    try:
        os.mkdir(os.path.join(inp,'slice','annotation'))
    except:
        pass
    files = glob.glob(os.path.join(inp,'*.tif'))
    for f in files:
        print(f)
        npy_img, npy_mask = multilayer_tif_2_single_layer_npy(f)
        split_image(npy_img, f)
        split_annotation(npy_mask, f)
        

if __name__=='__main__':
    inp = '/home/ins/Pictures/test76/Asphalt_Concrete_16.tif'
    im_dir = os.path.split(inp)[0]
    try:
        os.mkdir(os.path.join(im_dir,'slice'))
    except:
        pass
    try:
        os.mkdir(os.path.join(im_dir,'slice','image'))
    except:
        pass
    try:
        os.mkdir(os.path.join(im_dir,'slice','annotation'))
    except:
        pass
    npy_img, npy_mask = multilayer_tif_2_single_layer_npy(inp)
    split_image(npy_img, inp)
    split_annotation(npy_mask, inp)
