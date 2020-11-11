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

tk_root = None

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


def slice_for_train_folder():
    foder = askdirectory()
    if len(foder) == 0:
        return
    foder = os.path.normpath(foder)
    cls_name = tk_root.tkvar.get()
    answer = messagebox.askquestion("Satrt New Training ?", 
            'Yes to start new or No to resume')
    init_mode = 'new'
    if answer == 'no':
        init_mode = 'resume'
    start_subprocess(foder,init_mode,cls_name)
        
def select_raw_train_folder():
    foder = askdirectory()
    if len(foder) == 0:
        return
    foder = os.path.normpath(foder)
    init_mode = 'new'
    class_name = tk_root.tkvar.get()
    answer = messagebox.askquestion("Satrt New Training ?", 
            'Yes to start new or No to resume')
    init_mode = 'new'
    if answer == 'no':
        init_mode = 'resume'
    start_subprocess(foder,init_mode,class_name)

def start_subprocess(path,init_mode,cls_name):
    if os.name == 'nt':
        cmd_str1 = ['python', 'my_train.py', "%s"%path, init_mode, cls_name]
        print(cmd_str1)
        subprocess.Popen(cmd_str1, shell=True)
    else:
        subprocess.Popen(['python3 my_train.py "%s" %s %s'%(path, init_mode, cls_name)],shell=True)

def resume_training():
    fid = open('last_folder.txt','r')
    str1 = fid.read().rstrip()
    fid.close()
    fid = open('last_cls_name.txt','r')
    cls_name = fid.read()
    cls_name = cls_name.rstrip()
    fid.close()
    start_subprocess(str1,'resume',cls_name)

def build_page(root):
    global tk_root 
    tk_root = root

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

    #btn3 = Button(root, text="Select multi-layer TIF folder", 
    #        command=slice_tif_folder,
    #        height=1, 
    #        width=100, 
    #        font=('Helvetica', '18'))
    #btn3.pack(pady=(20,30))

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

    btn3 = Button(root, text="No slice", 
            command=select_raw_train_folder,
            height=1, 
            width=100, 
            font=('Helvetica', '18'))
    btn3.pack(pady=(10,30))
    btn3.config(image=photo0315, compound="left",
        width="400",
        height="60")

    choices = cfg.cls_list
    root.tkvar = StringVar(root)

    try:
        fid = open('last_cls_name.txt','r')
        cls_name = fid.read()
        cls_name = cls_name.rstrip()
    except:
        cls_name = 'Farmland'
        pass

    root.tkvar.set(cls_name)
    style = ttk.Style()
    style.configure('my.TMenubutton', font=('Arial', 30, 'bold'))
    root.popupMenu = OptionMenu(root, root.tkvar, *choices)
    root.popupMenu.config(width=12,font=('Helvetica',16))
    root.popupMenu.pack(pady=(10,40))
