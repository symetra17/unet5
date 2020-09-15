# This is a python3 code
# It’s amazing how the the human mind does not process the the fact I used the the word “the” twice each time in this sentence
import os
import shutil
from tkinter import Tk, PhotoImage
from PIL import Image
from PIL import ImageTk 
from tkinter import ttk

import page_train
import page_predict
import page_utils

#import getmac
#if getmac.get_mac_address() != "00:1b:21:bb:2c:72":
#    print("Terminated")
#    quit()

if __name__=='__main__':
    master = Tk()
    try:
        master.tk.call('tk_getOpenFile', '-foobarbaz')
    except:
        pass
    if os.name=='nt':
        pass
    else:
        master.tk.call('set', '::tk::dialog::file::showHiddenBtn', '1')
        master.tk.call('set', '::tk::dialog::file::showHiddenVar', '0')

    file_path = os.path.dirname(os.path.abspath(__file__)) 
    master.title('Insight A.I. Extraction Tool   ' + file_path)
    master.resizable(0,0)

    Image.MAX_IMAGE_PIXELS = None
    nb = ttk.Notebook(master)
    
    # adding Frames as pages for the ttk.Notebook
    # first page, which would get widgets gridded into it
    page1 = ttk.Frame(nb)
    page2 = ttk.Frame(nb)
    page3 = ttk.Frame(nb)
    nb.add(page1, text='\n  PREDICT  \n')
    nb.add(page2, text='\n  TRAINING  \n')
    nb.add(page3, text='\n  DATA UTILS \n')
    
    nb.grid(column=0)

    page_predict.build_page(page1)
    page_train.build_page(page2)
    page_utils.build_page(page3)

    file_path = os.path.dirname(os.path.abspath(__file__))
    tmp = os.path.join(file_path,'tmp')
    try:
        shutil.rmtree(tmp)
    except:
        pass
    #os.mkdir('tmp')
    master.mainloop()
