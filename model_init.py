import guicfg as cfg
import os
import time
import multiprocessing as mp

process_name = mp.current_process().name
if process_name != 'gen_shape':
    from keras import backend as K 
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

model = None
initalized = False
last_chkpt_path = None

def remove_ext(inp):
    return os.path.splitext(inp)[0]

def init(n_classes, bands, my_size, checkpoint_path=None):
    global initalized, model, last_chkpt_path
    from keras_segmentation.models.unet import unet, resnet50_unet
    if (not initalized) or (checkpoint_path != last_chkpt_path):
        try:
            del model
        except:
            pass
        K.clear_session()
        model = resnet50_unet(bands,
            n_classes = n_classes,
            input_height = my_size, 
            input_width  = my_size)
        latest_m = checkpoint_path
        model.load_weights(latest_m)
        last_chkpt_path = latest_m
        initalized = True
    
def do_prediction(fname):
    out = model.predict_segmentation(
        inp=fname,
        out_fname = remove_ext(fname) + "_result.bmp",
        out_fname2 = remove_ext(fname) + "_result2.bmp"
    )

def re_load(checkpoint_path):
    model.load_weights(checkpoint_path)

def destruct():
    global initalized, model, last_chkpt_path
    print('clearing session')

if __name__=='__main__':
    for n in range(5):
        init(3,5,512,R"C:\Users\echo\Code\unet5\weights\Squatter\vanilla_unet_1.weight")
        init(2,4,512,R"C:\Users\echo\Code\unet5\weights\Vehicles\vanilla_unet_1.weight")

    import keras
    keras.backend.clear_session()    

    import time
    print('done')
    time.sleep(6)