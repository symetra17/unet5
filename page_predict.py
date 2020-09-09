from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import os
from glob import glob
from PIL import ImageTk, Image 
from recombine import recombine, recombine2
import subprocess
import tkinter.ttk
from tkinter import messagebox
import platform
import model_init
import shutil
import cv2
import guicfg as cfg
import numpy as np
import math
import geotiff


tk_root = None
#my_size = cfg.my_size
flip_nir_red_ch = False

def path_insert_folde(filename, folder):
    splited = os.path.split(filename)
    return os.path.join(splited[0], folder, splited[1])

def remove_ext(inp):
    return os.path.splitext(inp)[0]

def replace_ext(inp, new_ext):
    result = os.path.splitext(inp)[0] + new_ext
    return result

def split_image(img, outname, my_size):
    new_folder = os.path.join(os.path.split(outname)[0], 'tmp')
    try:
        shutil.rmtree(new_folder)
    except:
        pass
    os.mkdir(new_folder)
    out_fname_list = []
    img_h, img_w = img.shape[:2]
    slice_visual = np.zeros((img_h,img_w,3),dtype=np.uint8)
    n_overlap = 40
    xstart_list = range(0, img_w-my_size, my_size-n_overlap)
    ystart_list = range(0, img_h-my_size, my_size-n_overlap)
    start_pos = []
    for xstart in xstart_list:
        xend   = xstart + my_size
        for ystart in ystart_list:
            yend   = ystart + my_size
            start_pos.append((xstart,ystart))
            cv2.rectangle(slice_visual,(xstart,ystart),(xend-1,yend-1),(255,255,255),1)
            sub_im = img[ystart:yend, xstart:xend, :]
            fullpath = remove_ext(outname) + '_W_%d_H_%d_X_%d_%d_Y_%d_%d.tif'%(img_w,img_h,
                    xstart,xend,ystart,yend)
            fullpath = path_insert_folde(fullpath, 'tmp')
            out_fname_list.append(fullpath)
            print('subim shape',sub_im.shape)
            geotiff.imwrite(fullpath, sub_im)
            
    cv2.imwrite('slice_visual.bmp',slice_visual)
    return out_fname_list, start_pos


def single_predict(fname, class_name):
    src_fname = fname
    down_scale = cfg.get(class_name).down_scale
    my_size = cfg.get(class_name).my_size
    bands = cfg.get(class_name).bands

    if down_scale > 1:
        im = geotiff.imread(fname)
        mul = 1/down_scale
        im = cv2.resize(im, None, fx=mul, fy=mul, interpolation=cv2.INTER_AREA)
        ds_fname = replace_ext(fname, '_dn_samp.tif')
        fname = ds_fname
        geotiff.imwrite(fname, im)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chk_point_path = os.path.join(current_dir, 'weights', class_name,'vanilla_unet_1.weight')
    if not os.path.exists(chk_point_path):
        messagebox.showinfo("Prediction terminated", "Weight file not found")
        return
    
    img = geotiff.imread(fname)
    
    new_height = my_size * math.ceil(img.shape[0]/my_size)
    new_width = my_size * math.ceil(img.shape[1]/my_size)
    img_pad = np.zeros((new_height, new_width, img.shape[2]), img.dtype)
    img_pad[0:img.shape[0], 0:img.shape[1], :] = img    
    slice_list,start_pos = split_image(img_pad, fname, my_size)
    for f in slice_list:
        model_init.init(bands, my_size, chk_point_path)
        model_init.do_prediction(f)
    
    recombine(fname, start_pos)
    res_file = recombine2(fname, start_pos)
        
    #import blur_bkgnd
    #blur_bkgnd.blur_bkgnd(fname,)
    #generate_tif_alpha

    if down_scale > 1:
        img = cv2.imread(res_file)
        img = cv2.resize(img,None,fx=down_scale,fy=down_scale)
        cv2.imwrite(res_file, img)

    pil_img = geotiff.imread(src_fname)

    print('src_fname', src_fname)
    if geotiff.is_geotif(src_fname):
        h = pil_img.shape[0]
        w = pil_img.shape[1]
        np4ch = np.zeros((h, w, 4), dtype=np.uint8)
        cp = img[0:h, 0:w, 0].copy()
        np4ch[:,:,3] = cp   # fill alpha channel with detection result
        np4ch[:,:,0] = cp   # fill 1st channel with detection result
        geotiff.generate_tif_alpha(np4ch, replace_ext(src_fname, '_geo.tif'),
                    src_fname)
        fff=replace_ext(src_fname, '_geo.tif')
        geotiff.polygonize(rasterTemp=fff,
                    outShp=replace_ext(src_fname, '_shape'))
    else:
        print('is not a geotiff')
        
def openimage():
    fname = askopenfilename(filetypes=(
        ('Image file', '*.jpg *.JPG *.jpeg *.JPEG *.tif *.bmp *.png'),
        ))
    if len(fname) == 0:
        return
    class_name = tk_root.tkvar.get()
    single_predict(fname, class_name)
    messagebox.showinfo("Prediction completed", "Prediction completed")


def openfolder_predict():
    fd = askdirectory()
    if len(fd) == 0:
        return
    files = []
    if os.name=='nt':
        ext_list = ['jpg', 'tif', 'png']
    else:
        ext_list = ['jpg', 'JPG', 'tif', 'TIF', 'png', 'PNG']
    for item in ext_list:
        files.extend(glob(os.path.join(fd,'*.'+item)))
    print(files)
    for fname in files:
        single_predict(fname)    
    messagebox.showinfo("Prediction completed", "Prediction completed")
    
def checkbox_callback():
    global flip_nir_red_ch
    flip_nir_red_ch = 'selected' in tk_root.checkbox1.state()

def build_page(root):
    global tk_root 
    tk_root = root

    lb = Label(root, text='Class Name')
    lb.pack(pady=(10,0))
    cfg_editbox = Text(root, height=2)
    cfg_editbox.pack(padx=5)
    tk_root.cfg_editbox = cfg_editbox
    import json
    tk_root.cfg_editbox.insert(END, json.dumps(cfg.classes_dict))

    var1 = IntVar()
    tk_root.checkbox1 = tkinter.ttk.Checkbutton(tk_root, text="NIR-RED CH SWAP", 
            command=checkbox_callback)
    tk_root.checkbox1.pack(pady=(30,0))
    tk_root.checkbox1.state(['!alternate'])

    btn0 = Button(root, text="  Select file", command=openimage, 
            height=1, width=80,  
            font=('Helvetica', '20'))
    btn0.pack(padx=(100,100), pady=(50,50))
    
    tk_root.photo11877 = PhotoImage(file=os.path.join('icon', "slicemagic.png"))
    btn0.config(image=tk_root.photo11877, compound="left", 
                height="60",
                width="400")
    
    btn1 = Button(root, text=" Select folder", command=openfolder_predict, 
            height=1, width=80, 
            font=('Helvetica', '20'))
    
    btn1.pack(pady=(10,10))
    tk_root.photo09589 = PhotoImage(file=os.path.join('icon', "predictf.png"))
    btn1.config(image=tk_root.photo09589,compound="left", 
                height="60",
                width="400")

    choices = ['Trees','Vehicles','Squatter','yyy','zzz']
    tk_root.tkvar = StringVar(tk_root)
    tk_root.tkvar.set('Squatter') # set the default option

    style = ttk.Style()
    style.configure('my.TMenubutton', font=('Arial', 30, 'bold'))

    tk_root.popupMenu = OptionMenu(tk_root, tk_root.tkvar, 
        *choices)

    tk_root.popupMenu.config(width=8,font=('Helvetica',16))
    tk_root.popupMenu.pack(pady=(10,10))
