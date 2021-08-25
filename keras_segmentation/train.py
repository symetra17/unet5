from multiprocessing import Process, current_process
import glob
import six
import inspect
import os,sys
import json
import multiprocessing as mp
import json_conv

if current_process().name=='MainProcess':
    from .data_utils.data_loader import image_segmentation_generator
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

def recut(train_src_dir, cls_name, subfolder='a'):
    json_conv.recut(train_src_dir, cls_name, subfolder)


def train(model, train_src_dir, input_height=None, input_width=None, n_classes=None, 
        verify_dataset=True, checkpoints_path=None, batch_size=4,
        auto_resume_checkpoint=False, load_weights=None, cls_name = '', do_augment=True):

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    if auto_resume_checkpoint:
        print("Loading the weights from latest checkpoint ")
        model.load_weights(checkpoints_path+'.weight')

    epochs = 10
    print('Re-cut training set')
    json_conv.recut(train_src_dir, cls_name, 'a')

    for ep in range(epochs):
    
        images_path = os.path.join(train_src_dir, 'slice'+'a', 'image')
        annots_path = os.path.join(train_src_dir, 'slice'+'a', 'annotation')
        p_cut = mp.Process(target=recut, args=(train_src_dir, cls_name, 'b', ))
        p_cut.start()
        train_gen = image_segmentation_generator(images_path, annots_path, batch_size, n_classes, input_height, input_width, output_height, output_width, do_augment=do_augment)
        files = glob.glob(os.path.join(train_src_dir,'slice'+'a','image','*.tif'))
        nfiles = len(files)
        steps_per_epoch = 1 + nfiles//batch_size
        print("Starting Epoch ", ep, "A")
        history_callback = model.fit_generator(train_gen, steps_per_epoch, epochs=2)
        model.save_weights(checkpoints_path + ".weight")
        p_cut.join()

        #--------------------------------------------------------------------------------------------------
        images_path = os.path.join(train_src_dir, 'slice'+'b', 'image')
        annots_path = os.path.join(train_src_dir, 'slice'+'b', 'annotation')
        p_cut = mp.Process(target=recut, args=(train_src_dir, cls_name, 'a', ))
        p_cut.start()
        train_gen = image_segmentation_generator(images_path, annots_path, batch_size, n_classes, input_height, input_width, output_height, output_width, do_augment=do_augment)
        files = glob.glob(os.path.join(train_src_dir,'slice'+'b','image','*.tif'))
        nfiles = len(files)
        steps_per_epoch = 1 + nfiles//batch_size
        print("Starting Epoch ", ep, "B")
        history_callback = model.fit_generator(train_gen, steps_per_epoch, epochs=2)
        model.save_weights(checkpoints_path + ".weight")
        p_cut.join()




#        loss_history = history_callback.history["loss"]
#        with open('train.log','a') as fid:
#            fid.write('epochs: %03d  '%ep)
#            fid.write('%.4f'%(loss_history[0]))
#            fid.write('\n')
