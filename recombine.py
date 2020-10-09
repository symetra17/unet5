import numpy as np
import cv2
import os
from glob import glob 
import PIL.Image
import guicfg as cfg
import geotiff

def remove_ext(inp):
    return os.path.splitext(inp)[0]

def path_insert_foldr(filename, folder):
    splited = os.path.split(filename)
    return os.path.join(splited[0], folder, splited[1])

def recombine(src_img_file, start_pos, output_folder):
    
    '''This function collects fragment image and forms a larger images'''
    #files = glob(path_insert_foldr(remove_ext(src_img_file)  + '*_result.tif', 'tmp'))
    files = glob(os.path.join('tmp', '*_result.tif'))

    www = files[0].split('_')
    file_head = '_'.join(www[0:-7])

    shape_src = geotiff.get_img_h_w(src_img_file)

    if not os.path.exists(files[0]):
        print(files[0]+'-------->file not exists')
        return
    my_size = geotiff.imread(files[0]).shape[0]   # this could be replace by geotiff.get_img_h_w() to speed up

    img_w = shape_src[1] + my_size
    img_h = shape_src[0] + my_size

    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    img_h, img_w = img.shape[:2]
    ntrim = 18
    for pos in start_pos:
        xstart = pos[0]
        ystart = pos[1]
        xend   = xstart + my_size
        yend   = ystart + my_size
        inp_name = file_head + '_X_%d_%d_Y_%d_%d_result.tif'%(xstart, xend, ystart, yend)
        sub_img = geotiff.imread(inp_name)
        img[ystart+ntrim:yend-ntrim, xstart+ntrim:xend-ntrim, :] = sub_img[ntrim:sub_img.shape[0]-ntrim,ntrim:sub_img.shape[1]-ntrim,:]

    # Determine final output image size, which should be equal to that of the source image
    crop_size = geotiff.get_img_h_w(src_img_file)
    # Crop result images to remove zeros padding on the right and bottom size of the image.
    outname = remove_ext(src_img_file) + '_result.tif'
    outname = os.path.join(output_folder, os.path.split(outname)[1])
    geotiff.imwrite(outname, img[0:crop_size[0], 0:crop_size[1], :])
    return

def recombine2(src_img_file, start_pos, output_folder):
    src_img_file = os.path.normpath(src_img_file)
    
    '''This function collects fragment image and forms a larger images'''
    #files = glob(path_insert_foldr(remove_ext(src_img_file)  + '*_result2.bmp', 'tmp'))
    files = glob(os.path.join('tmp', '*_result2.bmp'))
    www = files[0].split('_')
    file_head = '_'.join(www[0:-7])
    shape_src = geotiff.get_img_h_w(src_img_file)
    my_size = cv2.imread(files[0]).shape[0]
    img_w = shape_src[1] + my_size
    img_h = shape_src[0] + my_size + 1000
    img = np.zeros((img_h, img_w, 3) ,dtype=np.uint8)
    img_h, img_w = img.shape[:2]    
    for pos in start_pos:
        xstart = pos[0]
        ystart = pos[1]
        xend   = xstart + my_size
        yend   = ystart + my_size
        inp_name = file_head + '_X_%d_%d_Y_%d_%d_result2.bmp'%(xstart, xend, ystart, yend)
        sub_img = cv2.imread(inp_name)
        if sub_img is None:
            print('Could not read ', inp_name)
            quit()
        #img[ystart:yend, xstart:xend, :] = sub_img
        h = sub_img.shape[0]
        w = sub_img.shape[1]
        ntrim = 18
        img[ystart+ntrim:yend-ntrim, xstart+ntrim:xend-ntrim, :] = sub_img[ntrim:h-ntrim,ntrim:w-ntrim,:]

    # Determine final output image size, which should be equal to that of the source image
    crop_size = geotiff.get_img_h_w(src_img_file)
    # Crop result images to remove zeros padding on the right and bottom size of the image.
    h = img[0:crop_size[0], 0:crop_size[1], 0].copy()
    h[h>0] = 255
    ofname = remove_ext(src_img_file) + '_result_bw.bmp'
    #a_part = os.path.split(ofname)[0]
    #b_part = 'result'
    c_part = os.path.split(ofname)[1]
    #ofname = os.path.join(a_part,b_part,c_part)
    ofname = os.path.join(output_folder, c_part)
    cv2.imwrite(ofname, h)
    return ofname

if __name__=='__main__':
    recombine('/home/on99/Pictures/square_triangle/IMG_4011.jpg')

 