@echo on
call C:\Users\%USERNAME%\Anaconda3\Scripts\activate.bat 
call conda activate my_unet_env
python.exe gui.py
