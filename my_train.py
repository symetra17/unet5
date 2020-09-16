from keras_segmentation.models.unet import vgg_unet, unet
import guicfg as cfg
import sys
import os
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import datetime

if __name__=='__main__':
    
    if len(sys.argv) < 5:
        print('Requires input argument, the training folder and the annotation folder and initial mode(new/resume) clsname')
        quit()

    im_dir = sys.argv[1]
    ann_dir = sys.argv[2]
    arc = False
    if sys.argv[3] == 'resume':
        arc = True
    cls_name = sys.argv[4]
    n_classes = 1 + len(cfg.get(cls_name).cls_sub_list)
    my_size = cfg.get(cls_name).my_size
    bands = cfg.get(cls_name).bands

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default 

    fid = open('last_folder.txt','w')
    fid.write(im_dir + '\n')
    fid.write(ann_dir + '\n')
    fid.close()

    fid = open('last_cls_name.txt','w')
    fid.write(cls_name)
    fid.close()

    tstp = datetime.datetime.now()

    weight_dir = os.path.join('weights', cls_name)
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)

    fname = os.path.join(weight_dir, 'train.log')
    fid = open(fname,'w')
    fid.write('Training config log %s\n'%tstp)
    fid.write('Patch size:%d\n'%my_size)
    fid.write('Class name:%s\n'%cls_name)
    fid.write('N Band    :%d\n'%bands)
    fid.write('N Class   :%d\n'%n_classes)
    fid.write(im_dir + '\n')
    fid.write(ann_dir + '\n')
    fid.close()

    model = unet(bands,
            n_classes=n_classes,
            input_height=my_size, 
            input_width =my_size)
    model.train(
        train_images      = im_dir,
        train_annotations = ann_dir,
        checkpoints_path = os.path.join('weights', cls_name, 'vanilla_unet_1'),
        epochs = cfg.epochs,
        auto_resume_checkpoint=arc,
        steps_per_epoch = 512,
        batch_size = 4,
        cls_name = cls_name
    )
