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

def recombine(src_img_file, start_pos):
    
    '''This function collects fragment image and forms a larger images'''
    files = glob(path_insert_foldr(remove_ext(src_img_file)  + '*_result.tif', 'tmp'))
    www = files[0].split('_')
    file_head = '_'.join(www[0:-11])
    img_w, img_h = int(www[-10]), int(www[-8])
    
    if not os.path.exists(files[0]):
        print(files[0]+'-------->file not exists')
        return

    my_size = geotiff.imread(files[0]).shape[0]
    img = np.zeros((img_h, img_w, 3) ,dtype=np.uint8)
    img_h, img_w = img.shape[:2]
    ntrim = 18
    for pos in start_pos:
        xstart = pos[0]
        ystart = pos[1]
        xend   = xstart + my_size
        yend   = ystart + my_size
        inp_name = file_head + '_W_%d_H_%d_X_%d_%d_Y_%d_%d_result.tif'%(
                img_w, img_h, xstart, xend, ystart, yend)
        sub_img = geotiff.imread(inp_name)

        img[ystart+ntrim:yend-ntrim, xstart+ntrim:xend-ntrim, :] = sub_img[ntrim:sub_img.shape[0]-ntrim,ntrim:sub_img.shape[1]-ntrim,:]

    # Determine final output image size, which should be equal to that of the source image
    print('recombine',src_img_file)
    
    src_img = geotiff.imread(src_img_file)
    crop_size = (src_img.shape[0], src_img.shape[1])
    # Crop result images to remove zeros padding on the right and bottom size of the image.
    geotiff.imwrite(remove_ext(src_img_file) + '_result.tif', 
            img[0:crop_size[0], 0:crop_size[1], :])

    outname = ''
    if cfg.predict_output_format == 'jpg':
        outname = remove_ext(src_img_file) + '_result.jpg'
    elif cfg.predict_output_format == 'bmp':
        outname = remove_ext(src_img_file) + '_result.bmp'
    elif cfg.predict_output_format == 'png':
        outname = remove_ext(src_img_file) + '_result.png'

    return outname

def recombine2(src_img_file, start_pos):
    src_img_file = os.path.normpath(src_img_file)
    
    '''This function collects fragment image and forms a larger images'''
    files = glob(path_insert_foldr(remove_ext(src_img_file)  + '*_result2.bmp', 'tmp'))
    www = files[0].split('_')
    file_head = '_'.join(www[0:-11])
    img_w, img_h = int(www[-10]), int(www[-8])
    my_size = cv2.imread(files[0]).shape[0]
    img = np.zeros((img_h, img_w, 3) ,dtype=np.uint8)
    img_h, img_w = img.shape[:2]    
    for pos in start_pos:
        xstart = pos[0]
        ystart = pos[1]
        xend   = xstart + my_size
        yend   = ystart + my_size
        inp_name = file_head + '_W_%d_H_%d_X_%d_%d_Y_%d_%d_result2.bmp'%(
                img_w, img_h, xstart, xend, ystart, yend)
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
    src_img = geotiff.imread(src_img_file)
    crop_size = (src_img.shape[0], src_img.shape[1])
    # Crop result images to remove zeros padding on the right and bottom size of the image.

    #import geotiff
    #if geotiff.is_geotif(src_img_file):
    #    np4ch = np.zeros((pil_img.height, pil_img.width, 4), dtype=np.uint8)
    #    h = img[0:crop_size[0], 0:crop_size[1], 0].copy()
    #    h[h>0] = 255
    #    np4ch[:,:,3] = h   # fill alpha channel with detection result
    #    np4ch[:,:,0] = h   # fill 1st channel with detection result
    #    geotiff.generate_tif_alpha(np4ch, remove_ext(src_img_file) + '_result2.tif', 
    #                src_img_file)
    #    geotiff.polygonize(rasterTemp=remove_ext(src_img_file) + '_result2.tif', 
    #                outShp=remove_ext(src_img_file) + '_shape')
    #else:    
    #    h = img[0:crop_size[0], 0:crop_size[1], 0].copy()
    #    h[h>0] = 255
    #    cv2.imwrite(remove_ext(src_img_file) + '_result_bw.bmp', h)
            

    h = img[0:crop_size[0], 0:crop_size[1], 0].copy()
    h[h>0] = 255
    ofname = remove_ext(src_img_file) + '_result_bw.bmp'
    cv2.imwrite(ofname, h)
    return ofname

if __name__=='__main__':
    recombine('/home/on99/Pictures/square_triangle/IMG_4011.jpg')

 