# Simple slice

import cv2
import numpy as np
import os
import guicfg as cfg
import os
import shutil
from tkinter import Tk, PhotoImage, Button
from tkinter.filedialog import askopenfilename
USE_GUI = True
my_size = cfg.my_size
outext = 'png'

def remove_ext(inp):
    return os.path.splitext(inp)[0]

def path_insert_folde(filename, folder):
    splited = os.path.split(filename)
    return os.path.join(splited[0], folder, splited[1])

def split_image(img, outname):
    try:
        base_dir = os.path.split(outname)[0]
        os.mkdir(os.path.join(base_dir,'slice'))
    except:
        pass
    img_h, img_w = img.shape[:2]
    for n in range(int(img_w/my_size)):
        for m in range(int(img_h/my_size)):
            xstart = (n+0)*my_size
            xend   = (n+1)*my_size
            ystart = (m+0)*my_size
            yend   = (m+1)*my_size    
            sub_im = img[ystart:yend, xstart:xend, :]
            fullpath = remove_ext(outname) + '_W_%d_H_%d_X_%d_%d_Y_%d_%d.'%(img_w,img_h,xstart,xend,ystart,yend) + outext
            fpath = path_insert_folde(fullpath, 'slice')
            if fullpath[-3:] == 'jpg':
                cv2.imwrite(fpath, sub_im, [cv2.IMWRITE_JPEG_QUALITY, 98])
            else:
                cv2.imwrite(fpath, sub_im)

def select_single():
    fname = ''
    fname = askopenfilename()
    if len(fname) != 0:
        img=cv2.imread(fname)
        split_image(img, fname)

if __name__=='__main__':
    if USE_GUI:
        root = Tk()
        root.title('Insight Simple Slice Tool')

        try:
            root.tk.call('tk_getOpenFile', '-foobarbaz')
        except:
            pass
        root.tk.call('set', '::tk::dialog::file::showHiddenBtn', '1')
        root.tk.call('set', '::tk::dialog::file::showHiddenVar', '0')
        btn = Button(root, text="Select images", 
                command=select_single,
                height=5, 
                width=20,
                font=('Helvetica', '18'))
        btn.pack(padx=(10,10), pady=(10,10))
        root.mainloop()
    else: 
        inp='/home/ins/Pictures/mikania-3/DSC_0685.JPG'
        img = cv2.imread(inp)
        split_image(img,inp)
