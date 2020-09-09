# This is a python3 code
from tkinter import *
import os, sys, time, subprocess, random
import PIL
from PIL import Image
from PIL import ImageTk, Image 
import tkinter.filedialog as fdialog
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from tkinter import messagebox
from glob import glob
from tkinter import ttk
import cv2
import numpy as np
import guicfg as cfg
import time
import re
import json_conv
import multi_tif
import multiprocessing as mp

def path_insert_folde(filename, folder):
    splited = os.path.split(filename)
    return os.path.join(splited[0], folder, splited[1])

def remove_ext(inp):
    return os.path.splitext(inp)[0]

def clean_folder(folder):
    try:
        os.mkdir(folder)
    except:
        pass
    to_del = glob(os.path.join(folder,'*.*'))
    for f in to_del:
        os.remove(f)

def create_clean_folder(folder):
    try:
        os.mkdir(folder)
    except:
        pass
    clean_folder(folder)


def mp_split(files, cls_name):
    for fname in files:
        json_conv.split_for_training(fname, cls_name)

def slice_for_train_folder():
    foder = askdirectory()
    if len(foder) == 0:
        return
    foder = os.path.normpath(foder)
    if os.name == 'nt':
        ftypes = ['jpg', 'jpeg', 'tif', 'png', 'bmp']
    else:
        ftypes = ['jpg', 'JPG', 'jpeg', 'JPEG', 'tif', 'TIF', 'png', 'PNG', 'bmp', 'BMP']
    files = []
    for ft in ftypes:
        files.extend(glob(os.path.join(foder, '*.'+ft)))
    if len(files)==0:
        msg = 'No image input files found'
        print(msg)
        messagebox.showinfo("Training could not start", msg)
        return

    cls_name_list = json_conv.get_class_name(files)
    if len(cls_name_list) > 1:
        msg = "There are more than 1 class in labels"
        messagebox.showinfo("Training could not start", msg)
        return
    cls_name = cls_name_list[0]

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
    if n_thread > 2:
        n_thread = 2

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

    answer = messagebox.askquestion("Training dataset", 
        "Slicing completed, proceed ?")
    if answer == 'no':
        return

    answer = messagebox.askquestion("Satrt New Training ?", 'Yes to start new or No to resume')
    init_mode = 'new'
    if answer == 'no':
        init_mode = 'resume'
    start_subprocess(im_path,ann_path,init_mode,cls_name)
        
def slice_tif_folder():
    foder = askdirectory()
    if len(foder) == 0:
        return
    foder = os.path.normpath(foder)
    multi_tif.process_tif_dir(foder)
    answer = messagebox.askquestion("Training dataset",  '%s\nProceed ?'%foder)
    if answer == 'no':
        return

    im_path = os.path.join(foder, 'slice', 'image')
    ann_path = os.path.join(foder, 'slice', 'annotation')
    
    answer = messagebox.askquestion("Satrt New Training ?", 'Yes to start new or No to resume')
    init_mode = 'new'
    if answer == 'no':
        init_mode = 'resume'
    #start_subprocess(im_path,ann_path,init_mode,cls_name)

def select_raw_train_folder():
    foder = askdirectory()
    if len(foder) == 0:
        return
    foder = os.path.normpath(foder)
    im_path = os.path.join(foder, 'image')
    ann_path = os.path.join(foder, 'annotation')
    
    #answer = messagebox.askquestion("Satrt New Training ?", 'Yes to start new or No to resume')
    #init_mode = 'new'
    #if answer == 'no':
    #    init_mode = 'resume'
    #start_subprocess(im_path,ann_path,init_mode,)

def start_subprocess(im_path,ann_path,init_mode,cls_name):
    if os.name == 'nt':
        cmd_str1 = ['python', 'my_train.py', "%s"%im_path, "%s"%ann_path, init_mode, cls_name]
        subprocess.Popen(cmd_str1, shell=True)
    else:
        subprocess.Popen(['python3 my_train.py "%s" "%s" %s %s'%(im_path, ann_path, init_mode, cls_name)],shell=True)

def resume_training():
    fid = open('last_folder.txt','r')
    str1 = fid.read()
    fid.close()
    fid = open('last_cls_name.txt','r')
    cls_name = fid.read()
    cls_name = cls_name.rstrip()
    fid.close()
    paths = str1.split('\n')
    im_path = paths[0].rstrip()
    ann_path = paths[1].rstrip()
    start_subprocess(im_path,ann_path,'resume',cls_name)

def build_page(root):
    # Pre-tranning slice
    
    str1 = "\nThe training image should comes with a label file with same base filename and json file extension"
    lb = Label(root, wraplength=450, 
        text=str1)
    lb.pack()

    btn = Button(root, text="Select JSON folder", 
            command=slice_for_train_folder,
            height=1, 
            width=100, 
            font=('Helvetica', '18'))
    btn.pack(pady=(20,30))

    btn3 = Button(root, text="Select multi-layer TIF folder", 
            command=slice_tif_folder,
            height=1, 
            width=100, 
            font=('Helvetica', '18'))
    btn3.pack(pady=(20,30))

    btn2 = Button(root, text=" Resume", 
            command=resume_training,
            height=1, 
            width=100, 
            font=('Helvetica', '18'))
    btn2.pack(pady=(10,30))


    global photo03485
    photo03485=PhotoImage(file=os.path.join('icon', "open.png"))
    
    global photo0315
    photo0315=PhotoImage(file=os.path.join('icon', "train.png"))

    root.photo0450=PhotoImage(file=os.path.join('icon', "resume.png"))

    btn.config(image=photo0315, compound="left",
        width="400",
        height="60")
    btn2.config(image=root.photo0450, compound="left",
        width="400",
        height="60")
    btn3.config(image=photo03485, compound="left",
        width="400",
        height="60")

    btn3 = Button(root, text="No slice", 
            command=select_raw_train_folder,
            height=1, 
            width=100, 
            font=('Helvetica', '18'))
    btn3.pack(pady=(10,30))
    btn3.config(image=photo0315, compound="left",
        width="400",
        height="60")
