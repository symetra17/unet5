#!/usr/bin/env python
# coding: utf-8

# # Select by Area GUI

# In[1]:


from Filter_SHP_By_Area_v2 import *
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import os
import re
# from osgeo import ogr
# import shapefile


def validateTitle(title):
    rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
    new_title = re.sub(rstr, "_", title)  # replace with _
    return new_title


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


# In[21]:


window = Tk()

window.title("Select Features by Area")

lb1 = Label(window, text="Input SHP files path: ", font=("Arial", 10))
lb1.grid(column = 0, row = 1)

lb2 = Label(window, text ="", font=("Arial", 10))
lb2.grid(column = 1, row = 1)

lb3 = Label(window, text="Output SHP files path: ", font=("Arial", 10))
lb3.grid(column = 0, row = 4)

lb4 = Label(window, text ="", font=("Arial", 10))
lb4.grid(column = 1, row = 4)

lb5 = Label(window, text="Area threshold (sqr meter): ", justify=LEFT, font=("Arial", 10))
lb5.grid(column = 0, row = 6)

txt1 = Entry(window,text ="", width=6)
txt1.grid(column=1, row=6)

lb6 = Label(window, text ="92.9", font=("Arial", 10))
lb6.grid(column = 2, row = 6)

# lb9 = Label(window, text ="92.9", font=("Arial", 10))
# lb9.grid(column = 3, row = 6)

lb7 = Label(window, text ="The Suffix for output is:", font=("Arial", 10))
lb7.grid(column = 0, row = 8)

lb8 = Label(window, text ="", font=("Arial", 10))
lb8.grid(column = 2, row = 8)

txt2 = Entry(window,text = "", width=6)
txt2.grid(column=1, row=8)

def clicked():
    answer = filedialog.askdirectory(parent = window,
                                 initialdir = os.getcwd(),
                                 title = "select the folder stores SHP files:")
    lb2.configure(text = answer)
    
    
def clicked2():
    answer = filedialog.askdirectory(parent = window,
                                 initialdir = os.getcwd(),
                                 title = "Select the output folder:")
    lb4.configure(text = answer)

    
def clicked3():
    res = txt1.get()
    if is_number(res):
        lb6.configure(text = res)
    else:
        messagebox.showinfo('Error', 'Please fill a number!')
        lb6.configure(text = '92.9')

def clicked5():
    res = txt2.get()
    vres = validateTitle(res)
    lb8.configure(text = vres)

    
def clicked4():
    current_path = lb2['text']
    output_path = lb4['text']
    area_threshold = float(lb6['text'])
    Suffix = lb8['text']
    
    if current_path =="" or output_path =="":
        messagebox.showinfo('Error', 'Please fill the correct path for input and output!')
        
    elif (current_path !="") & (current_path == output_path) & (Suffix == ""):
        messagebox.showinfo('Error', 'Please fill a different path for output or change Suffix!')
        
    else:
        batch_select_shp (current_path, output_path, area_threshold, Suffix) #(current_path, output_path, tfw_path)
        messagebox.showinfo('Progress Information', 'Data Processing Finished!')
    

btn1 = Button(window, text="Select the SHP files folder", width=25, command = clicked)
btn1.grid(column = 3, row = 1)

btn2 = Button(window, text="Select the output folder", width=25, command = clicked2)
btn2.grid(column = 3, row = 4)      

btn3 = Button(window, text="Set the Area threshold (Optional)", width=25, command = clicked3)
btn3.grid(column = 3, row = 6)

btn5 = Button(window, text="Set Suffix for output (Optional)", width=25, command = clicked5)
btn5.grid(column = 3, row = 8)

btn4 = Button(window, text="Run", width=10, font=("Arial Bold", 12), bg="white", fg ='blue', command = clicked4)
btn4.grid(column = 4, row = 9) 

window.mainloop()


# In[ ]:





# In[ ]:




