#!/usr/bin/env python
# coding: utf-8




import BatchModelAccuracy as bmal
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import os




window = Tk()

window.title("Model's Accuracy")

lb1 = Label(window, text="BMP file path: ", font=("Arial", 10))
lb1.grid(column = 0, row = 1)

lb2 = Label(window, text =" ", font=("Arial", 10))
lb2.grid(column = 1, row = 1)

lb3 = Label(window, text="JSON file path: ", font=("Arial", 10))
lb3.grid(column = 0, row = 4)

lb4 = Label(window, text =" ", font=("Arial", 10))
lb4.grid(column = 1, row = 4)

lb5 = Label(window, text="Output file path: ", justify=LEFT, 
            font=("Arial", 10))
lb5.grid(column = 0, row = 6)

lb6 = Label(window, text =" ", font=("Arial", 10))
lb6.grid(column = 1, row = 6)

def clicked():
    answer = filedialog.askdirectory(parent = window,
                                 initialdir = os.getcwd(),
                                 title = "select the folder stores BMP files:")
    lb2.configure(text = answer)
    

    
def clicked2():
    answer = filedialog.askdirectory(parent = window,
                                 initialdir = os.getcwd(),
                                 title = "Select the folder stores JSON files:")
    lb4.configure(text = answer)

    
def clicked3():
    answer = filedialog.askdirectory(parent = window,
                                 initialdir = os.getcwd(),
                                 title = "Select the folder to store the Calculation Result:")
    lb6.configure(text = answer)
    
def clicked4():
    detect_folder = lb2['text']
    json_folder = lb4['text']
    output_path = lb6['text']
    bmal.batch_cal(json_folder, detect_folder,output_path)
    messagebox.showinfo('Progress Information', 'Calculation Finished!')
    

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

window.mainloop()





