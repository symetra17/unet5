from keras_segmentation.models.unet import vgg_unet, unet
import guicfg as cfg
import sys
import os
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

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

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default 

    fid = open('last_folder.txt','w')
    fid.write(im_dir)
    fid.write('\n')
    fid.write(ann_dir)
    fid.write('\n')
    fid.close()

    fid = open('last_cls_name.txt','w')
    fid.write(cls_name)
    fid.close()

    my_size = cfg.get(cls_name).my_size
    bands = cfg.get(cls_name).bands
    model = unet(bands,
            n_classes=2,
            input_height=my_size, 
            input_width =my_size)
    model.train(
        train_images      = im_dir,
        train_annotations = ann_dir,
        checkpoints_path = os.path.join('weights', cls_name, 'vanilla_unet_1'),
        epochs = cfg.epochs,
        auto_resume_checkpoint=arc,
        steps_per_epoch = 512,
        batch_size=3
    )
