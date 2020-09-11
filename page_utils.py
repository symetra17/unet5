from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import os
from glob import glob
from PIL import ImageTk, Image 
import subprocess
import tkinter.ttk
from tkinter import messagebox
import platform
import shutil
import cv2
import guicfg as cfg
import numpy as np
import math
import dataset_utils
import delete_blank

global tk_root 

def foo():
    fd = askdirectory(title=R'Choose "Slice" directory')
    if len(fd) == 0:
        return
    str1 = 'Non-blank/blank ratio: %.2f'%dataset_utils.get_statistics(fd)
    messagebox.showinfo("Dataset stats", str1)

def remove_blank(fd, portion):
    ratio = delete_blank.delete_blank(fd, portion)
    str1 = 'Non-blank/blank ratio: %.2f'%ratio
    messagebox.showinfo("Dataset stats", str1)

def btn1_callback():
    fd = askdirectory(title=R'Choose "Slice" directory')
    if len(fd) == 0:
        return
    remove_blank(fd, 0.2)

def btn2_callback():
    fd = askdirectory(title=R'Choose "Slice" directory')
    if len(fd) == 0:
        return
    remove_blank(fd, 0.4)

def build_page(root):
    global tk_root 
    tk_root = root

    btn0 = Button(root, text=R" Get Blank/Non Blank Stats", command=foo, 
            height=1,
            width=100,  
            font=('Helvetica', '16'))
    btn0.pack(padx=(100,100), pady=(40,40))

    tk_root.photo092 = PhotoImage(file=os.path.join('icon', "slicemagic.png"))
    btn0.config(image=tk_root.photo092, compound="left", 
                height="60",
                width="400")

    btn1 = Button(root, text=R" Remove 20% blank images ", command=btn1_callback, 
            height=1,
            width=100,  
            font=('Helvetica', '16'))
    btn1.pack(padx=(100,100), pady=(40,40))
    tk_root.photo_098 = PhotoImage(file=os.path.join('icon', "trim.png"))
    btn1.config(image=tk_root.photo_098, compound="left", 
                height="60",
                width="400")

    btn1 = Button(root, text=R" Remove 40% blank images ", command=btn2_callback, 
            height=1,
            width=100,  
            font=('Helvetica', '16'))
    btn1.pack(padx=(100,100), pady=(40,40))
    tk_root.photo_099 = PhotoImage(file=os.path.join('icon', "trim.png"))
    btn1.config(image=tk_root.photo_099, compound="left", 
                height="60",
                width="400")
