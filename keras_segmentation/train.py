import argparse
import json
from .data_utils.data_loader import image_segmentation_generator, \
    verify_segmentation_dataset
import os,sys
import glob
import six
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import json_conv
from pathlib import Path

def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f).isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid".format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files, key=lambda f: int(get_epoch_number_from_path(f)))
    return latest_epoch_checkpoint


def train(model,
          train_src_dir,          # image and annotation 
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          cls_name = '',
          do_augment=True
          ):

    from .models.all_models import model_from_name
    # check if user gives model name instead of the model object
    if isinstance(model, six.string_types):
        # create the model from the name
        assert (n_classes is not None), "Please provide the n_classes"
        if (input_height is not None) and (input_width is not None):
            model = model_from_name[model](
                n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)


    for ep in range(epochs):
        print("Starting Epoch ", ep)

        print('Re-cut training set')
        json_conv.xxx(train_src_dir, cls_name, 'a')

        train_images = os.path.join(train_src_dir,'slice'+'a', 'image')
        train_annotations = os.path.join(train_src_dir, 'slice'+'a', 'annotation')

        train_gen = image_segmentation_generator(
            train_images, train_annotations,  batch_size,  n_classes,
            input_height, input_width, output_height, output_width,do_augment=do_augment)
        files = glob.glob(os.path.join(train_src_dir,'slice'+'a','image','*.tif'))
        nfiles = len(files)
        steps_per_epoch = 1 + nfiles//batch_size
        history_callback = model.fit_generator(train_gen, steps_per_epoch, epochs=2)

        loss_history = history_callback.history["loss"]
        fid=open('train.log','a')
        fid.write('epochs: %03d  '%ep)
        fid.write('%.4f'%(loss_history[0]))
        fid.write('\n')
        fid.close()

        model.save_weights(checkpoints_path + ".weight")            
        if (ep+1)%4 == 0:
            print('Saving weight')
            model.save_weights(checkpoints_path + "." + str(ep))
    
