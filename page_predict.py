from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import os
from pathlib import Path
import shutil
from glob import glob
from PIL import ImageTk, Image 
import subprocess
import cv2
import gdal
import numpy as np
import math
import tkinter.ttk
from tkinter import messagebox

from recombine import recombine, recombine2
import platform
import model_init
import guicfg as cfg
import geotiff
import shp_filter
import threading

tk_root = None
dom_files = []
dsm_files = []
dsm_folder = None

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
            geotiff.imwrite(fullpath, sub_im)            
    #cv2.imwrite('slice_visual.bmp',slice_visual)
    return out_fname_list, start_pos


def single_predict(fname, class_name, fname_dsm=None):
    src_fname = fname
    down_scale = cfg.get(class_name).down_scale
    my_size = cfg.get(class_name).my_size
    bands = cfg.get(class_name).bands
    n_sub_cls = 1 + len(cfg.get(class_name).cls_sub_list)

    if down_scale > 1:
        im = geotiff.imread(fname)
        assert len(im.shape) > 2        # color
        assert im.shape[2] == 4         # 4 channels
        if fname_dsm is not None:
            dsm = geotiff.imread(fname_dsm)
            assert len(dsm.shape) == 2
            dsm = np.expand_dims(dsm, axis=2)
            im = np.concatenate((im, dsm), axis=2)
            assert im.shape[2] == 5         # 5 channels

        mul = 1/down_scale
        im = cv2.resize(im, None, fx=mul, fy=mul, interpolation=cv2.INTER_AREA)
        ds_fname = replace_ext(fname, '_dn_samp.tif')
        #fname1 = os.path.splitext(fname)[0] + '_dn_samp.tif'
        #fname = str(Path(Path(fname1).parent, 'result', Path(fname1).name))
        fname = ds_fname
        geotiff.imwrite(fname, im)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chk_point_path = Path(current_dir, 'weights', class_name, 'vanilla_unet_1.weight')
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
        model_init.init(n_sub_cls, bands, my_size, str(chk_point_path))
        model_init.do_prediction(f)
    cv2.destroyWindow('Predict')
    recombine(fname, start_pos)
    res_file = recombine2(fname, start_pos)

    img = cv2.imread(res_file)
    if down_scale > 1:
        img = cv2.resize(img,None,fx=down_scale,fy=down_scale)
        cv2.imwrite(res_file, img)

    if geotiff.is_geotif(src_fname):
        npa = geotiff.imread(src_fname)
        src_h = npa.shape[0]
        src_w = npa.shape[1]

        np4ch = np.zeros((src_h, src_w, 4), dtype=np.uint8)
        cp = img[0:src_h, 0:src_w, 0].astype(np.uint8).copy()
        np4ch[:,:,3] = cp   # fill alpha channel with detection result
        np4ch[:,:,0] = cp   # fill 1st channel with detection result
        out_path = Path(replace_ext(src_fname, '_result_geo.tif'))
        out_path = Path(out_path.parent,'result',out_path.name)
        geotiff.generate_tif_alpha(np4ch, str(out_path), src_fname)

        result_mask_path = out_path
        out_shp_file = Path(os.path.splitext(src_fname)[0] + '_shape')
        out_shp_file = Path(out_shp_file.parent, 'result', Path(out_shp_file.name))
        geotiff.polygonize(rasterTemp=str(result_mask_path), outShp=str(out_shp_file))
        shp_filter.add_area_single(str(out_shp_file/'predicted_object.shp'), 
                10, out_shp_file/'filtered'/'predicted_object.shp', class_name)
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
    class_name = tk_root.tkvar.get()
    for fname in files:
        single_predict(fname, class_name)
    messagebox.showinfo("Prediction completed", "Prediction completed")

def write_table():
    if len(dom_files) != len(dsm_files):
        if dsm_folder is not None:
            messagebox.showinfo("Input error", "Number of file unmatch")

    for n in range(len(dom_files)):
        fname = dom_files[n]
        fname = os.path.normpath(fname)
        fname_body = os.path.split(fname)[-1]
        datafile = gdal.Open(fname, gdal.GA_ReadOnly)
        cols = datafile.RasterXSize
        rows = datafile.RasterYSize
        bands = datafile.RasterCount
        geoinformation = datafile.GetGeoTransform()
        tk_root.tree.insert("",END,text='(DOM) '+fname_body, tags=('oddrow',), 
                        value=(cols, rows, bands, geoinformation))

        try:
            fname = dsm_files[n]
            fname = os.path.normpath(fname)
            fname_body = os.path.split(fname)[-1]
            datafile = gdal.Open(fname, gdal.GA_ReadOnly)
            cols = datafile.RasterXSize
            rows = datafile.RasterYSize
            bands = datafile.RasterCount
            geoinformation = datafile.GetGeoTransform()
            tk_root.tree.insert("",END,text=('(DSM) ') + fname_body, tags=('oddrow',),
                        value=(cols, rows, bands, geoinformation))
        except:
            print('number of file unmatch')
            #messagebox.showinfo("Input error", "Number of file unmatch")


def select_dom_folder():
    global dom_folder
    global dom_files
    fd = askdirectory()
    if len(fd) == 0:
        return
    dom_folder = fd
    files = glob(os.path.join(fd,'*.tif'))

    tk_root.entry1.delete(0, END)
    tk_root.entry1.insert(10, dom_folder)

    for i in tk_root.tree.get_children():
        tk_root.tree.delete(i)

    #datafile = gdal.Open(files[0], gdal.GA_ReadOnly)
    #print(dir(datafile))
    files.sort()
    dom_files = []
    for fname in files:
        if '_dn_samp' not in fname:
            dom_files.append(fname)
    write_table()


def select_dsm_folder():
    global dsm_folder
    fd = askdirectory()
    if len(fd) == 0:
        return
    dsm_folder = fd
    tk_root.entry2.delete(0, END)
    tk_root.entry2.insert(10, dsm_folder)
    for i in tk_root.tree.get_children():
        tk_root.tree.delete(i)

    global dsm_files
    dsm_files = glob(os.path.join(fd,'*.tif'))
    dsm_files.sort()
    write_table()

def predict_thread():
    try:
        os.mkdir(os.path.join(dom_folder,'result'))
    except:
        pass
    class_name = tk_root.tkvar.get()
    if class_name == 'Squatter':
        assert len(dom_files) == len(dsm_files)
        for n, dom_file in enumerate(dom_files):
            dsm_file = dsm_files[n]
            single_predict(dom_file, class_name, dsm_file)
    else:
        for dom_file in dom_files:
                single_predict(dom_file, class_name)
    folder = os.path.join(dom_folder,'tmp')
    shutil.rmtree(folder)
    tk_root.btn_start['state'] = 'normal'
    messagebox.showinfo("Prediction completed", "Prediction completed\n\n\n")


def start_predict():    
    tk_root.btn_start['state'] = 'disable'
    #thd1 = threading.Thread(target=predict_thread)
    #thd1.start()
    predict_thread()

def menu_callback(event):
    if event == 'Squatter':
        tk_root.btn2['state'] = 'normal'
        tk_root.entry2['state'] = 'normal'
    else:
        tk_root.btn2['state'] = 'disable'
        tk_root.entry2['state'] = 'disable'

def build_page(root):
    global tk_root 
    tk_root = root

    root.left_frame1 = Frame(root)
    root.left_frame1.pack(side=LEFT)

    root.frame1 = Frame(root.left_frame1)
    root.frame1.pack()

    root.text = Label(root.frame1, text='Class Name')
    root.text.pack(side=LEFT)

    #lb = Label(root, text='Class Name')
    #lb.pack(pady=(10,0))
    #cfg_editbox = Text(root, height=2)
    #cfg_editbox.pack(padx=5)
    #tk_root.cfg_editbox = cfg_editbox
    #import json
    #tk_root.cfg_editbox.insert(END, json.dumps(cfg.classes_dict))

    choices = cfg.cls_list
    tk_root.tkvar = StringVar(tk_root)
    tk_root.tkvar.set('Farmland') # set the default option
    style = ttk.Style()
    style.configure('my.TMenubutton', font=('Arial', 30, 'bold'))
    tk_root.popupMenu = OptionMenu(root.frame1, tk_root.tkvar, *choices, command=menu_callback)
    #tk_root.popupMenu.config(width=8,font=('Helvetica',16))
    tk_root.popupMenu.pack(pady=(10,10))

    btn_size = 350

    btn0 = Button(root.left_frame1, text="  Select file", command=openimage, 
            height=1, width=btn_size,  
            font=('Helvetica', '20'))
    #btn0.pack(padx=(30,30), pady=(20,20))
    
    tk_root.photo11877 = PhotoImage(file=os.path.join('icon', "slicemagic.png"))
    btn0.config(image=tk_root.photo11877, compound="left", 
                height="60",
                width=btn_size)
    
    root.entry1 = Entry(root.left_frame1, width=60)
    root.entry1.pack(pady=(20,5))

    btn1 = Button(root.left_frame1, text=" Select DOM folder", command=select_dom_folder, 
            height=1, width=btn_size, 
            font=('Helvetica', '20'))
    
    btn1.pack(padx=(30,30), pady=(10,10))
    tk_root.photo09589 = PhotoImage(file=os.path.join('icon', "DOM.png"))
    btn1.config(image=tk_root.photo09589,compound="left", 
                height="60",
                width=btn_size)


    root.entry2 = Entry(root.left_frame1, width=60)
    root.entry2.pack(pady=(20,5))
    btn2 = Button(root.left_frame1, text=" Select DSM folder", command=select_dsm_folder, 
            height=1, width=btn_size, 
            font=('Helvetica', '20'))    
    btn2.pack(pady=(10,10))
    root.btn2 = btn2
    root.btn2['state'] = 'disable'

    tk_root.photo619 = PhotoImage(file=os.path.join('icon', "DSM_sel.png"))
    btn2.config(image=tk_root.photo619,compound="left", 
                height="60",
                width=btn_size)
    root.entry2['state'] = 'disable'

    tk_root.btn_start = Button(root.left_frame1, text="   Start  ", command=start_predict, 
            height=1, width=btn_size, 
            font=('Helvetica', '20'))    
    tk_root.btn_start.pack(pady=(10,30))
    tk_root.photo064 = PhotoImage(file=os.path.join('icon', "start.png"))
    tk_root.btn_start.config(image=tk_root.photo064,compound="left", 
                height="60", width=btn_size)

    root.tree = ttk.Treeview(root, height=28)
    root.tree.pack(side=RIGHT, padx=(20,20))
    root.tree["columns"]=("one","two","three", "four")

    root.tree.column("#0", width=300, minwidth=300, stretch=NO)
    root.tree.column("one", width=50, minwidth=50, stretch=NO)
    root.tree.column("two", width=50, minwidth=50)
    root.tree.column("three", width=80, minwidth=50, stretch=NO)
    root.tree.column("four", width=80, minwidth=50, stretch=NO)

    root.tree.heading("#0",text="Name",anchor=W)
    root.tree.heading("one", text="Width",anchor=W)
    root.tree.heading("two", text="Height",anchor=W)
    root.tree.heading("three", text="No. Bands",anchor=W)
    root.tree.heading("four", text="Geo Info",anchor=W)

    #folder1=tree.insert("", 1, "", text="Folder 1", values=("23-Jun-17 11:05","File folder",""))
    #tree.insert("", 2, "", text="text_file.txt", values=("23-Jun-17 11:25","TXT file","1 KB"))
    
    root.tree.tag_configure('oddrow', background='orange')
    