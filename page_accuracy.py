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
import BatchModelAccuracy as bmal

global tk_root 

def clicked():
    answer = filedialog.askdirectory(parent=tk_root,
                                 initialdir=os.getcwd(),
                                 title="Select BMP folder:")
    tk_root.lb2.configure(text=answer)

def clicked2():
    answer = filedialog.askdirectory(parent=tk_root,
                                 initialdir=os.getcwd(),
                                 title="Select JSON folder:")
    tk_root.lb4.configure(text=answer)

def clicked3():
    answer = filedialog.askdirectory(parent=tk_root,
                                 initialdir=os.getcwd(),
                                 title="Select result folder:")
    tk_root.lb6.configure(text=answer)

def clicked4():
    detect_folder = tk_root.lb2['text']
    json_folder = tk_root.lb4['text']
    output_path = tk_root.lb6['text']
    bmal.batch_cal(json_folder, detect_folder,output_path)
    messagebox.showinfo('Progress Information', 'Calculation finished')


def build_page(root):
    global tk_root
    tk_root = root
    window = root
    tk_root.lb1 = Label(window, text="BMP file path: ", font=("Arial", 10))
    tk_root.lb1.grid(column = 0, row = 1)
    tk_root.lb2 = Label(window, text =" ", font=("Arial", 10))
    tk_root.lb2.grid(column = 1, row = 1)
    tk_root.lb3 = Label(window, text="JSON file path: ", font=("Arial", 10))
    tk_root.lb3.grid(column = 0, row = 4)
    tk_root.lb4 = Label(window, text =" ", font=("Arial", 10))
    tk_root.lb4.grid(column = 1, row = 4)
    tk_root.lb5 = Label(window, text="Output file path: ", justify=LEFT, 
                font=("Arial", 10))
    tk_root.lb5.grid(column = 0, row = 6)
    tk_root.lb6 = Label(window, text =" ", font=("Arial", 10))
    tk_root.lb6.grid(column = 1, row = 6)

    btn1 = Button(window, text="Select the folder contains detection result (.bmp)", 
              command = clicked)
    btn1.grid(column = 0, row = 0)

    btn2 = Button(window, text="Select the folder contains the tags (.json)", 
              command = clicked2)
    btn2.grid(column = 0, row = 3)      

    btn3 = Button(window, text="Select the folder to store the calculation result", 
              command = clicked3)
    btn3.grid(column = 0, row = 5)

    btn4 = Button(window, text="Start Calculation", 
              command = clicked4)
    btn4.grid(column = 2, row = 8) 
