import guicfg as cfg
import os
#import tensorflow as tf

from keras_segmentation.models.unet import unet


#from keras_segmentation.train import find_latest_checkpoint
#from keras.backend.tensorflow_backend import set_session

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#config = tf.ConfigProto()
#config = tf.compat.v1.ConfigProto() 
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = False  # to log device placement (on which device the operation ran)
##sess = tf.Session(config=config)
#sess = tf.compat.v1.Session()
#set_session(sess)  # set this TensorFlow session as the default 

model = None
initalized = False
last_chkpt_path = None

def remove_ext(inp):
    return os.path.splitext(inp)[0]


def init(bands, my_size, checkpoint_path=None):

    global initalized, model, last_chkpt_path
    if not initalized:
        model = unet(bands,
            n_classes = 1 + len(cfg.classes_dict),  
            input_height = my_size, 
            input_width  = my_size)
        latest_m = checkpoint_path
        model.load_weights(latest_m)
        last_chkpt_path = latest_m
        initalized = True
    else:
        if checkpoint_path != last_chkpt_path:
            # Reload weight file, in case we do another prediction in using 
            # different weight file
            model.load_weights(checkpoint_path)
    
def do_prediction(fname):
    out = model.predict_segmentation(
        inp=fname,
        out_fname = remove_ext(fname) + "_result.bmp",
        out_fname2 = remove_ext(fname) + "_result2.bmp"
    )

def re_load(checkpoint_path):
    model.load_weights(checkpoint_path)

if __name__=='__main__':
    init(512,R"C:\Users\echo\Code\unet5\weights\Squatter\vanilla_unet_1.weight")

