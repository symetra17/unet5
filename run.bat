@echo on
call C:\Users\%USERNAME%\Anaconda3\Scripts\activate.bat 
call conda activate unet5_env
python.exe gui.py
