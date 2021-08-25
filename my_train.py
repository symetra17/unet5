import warnings
warnings.filterwarnings("ignore")
import guicfg as cfg
import sys
import os
import datetime

from multiprocessing import Process, current_process
if current_process().name=='MainProcess':
    from keras_segmentation.models.unet import vgg_unet, unet, resnet50_unet
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__=='__main__':

    if len(sys.argv) < 4:
        print('Requires input argument, [training folder] [initial mode(new/resume)] [cls_name]')
        quit()

    src_dir = sys.argv[1]
    arc = False
    if sys.argv[2] == 'resume':
        arc = True
    cls_name = sys.argv[3]
    n_classes = 1 + len(cfg.get(cls_name).cls_sub_list)
    my_size = cfg.get(cls_name).my_size
    bands = cfg.get(cls_name).bands

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default 

    fid = open('last_folder.txt','w')
    fid.write(src_dir)
    fid.close()

    fid = open('last_cls_name.txt','w')
    fid.write(cls_name)
    fid.close()

    tstp = datetime.datetime.now()

    weight_dir = os.path.join('weights', cls_name)
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)

    fname = 'train.log'
    fid = open(fname,'a')
    fid.write('Training config log %s\n'%tstp)
    fid.write('Patch size:%d\n'%my_size)
    fid.write('Class name:%s\n'%cls_name)
    fid.write('N Band    :%d\n'%bands)
    fid.write('N Class   :%d\n'%n_classes)
    fid.write(src_dir + '\n')
    fid.close()

    model = resnet50_unet(bands, n_classes=n_classes, input_height=my_size, input_width =my_size)
    model.train(train_src_dir=src_dir, checkpoints_path=os.path.join('weights', cls_name, 'vanilla_unet_1'), auto_resume_checkpoint=arc, cls_name=cls_name, do_augment=cfg.augm_flip)

#python my_train.py C:\Users\dva\unet5\weights\Vehicles new Vehicles