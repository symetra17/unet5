import glob
import random
import json
import os
import geotiff
import cv2
import numpy as np
from tqdm import tqdm
from keras.models import load_model

from .train import find_latest_checkpoint
from .data_utils.data_loader import get_image_array, get_segmentation_array, DATA_LOADER_SEED, class_colors , get_pairs_from_paths
from .models.config import IMAGE_ORDERING
from . import metrics

import six
import guicfg as cfg
import cfg_color

random.seed(DATA_LOADER_SEED)

def model_from_checkpoint_path(checkpoints_path):

    from .models.all_models import model_from_name
    assert (os.path.isfile(checkpoints_path+"_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(
        open(checkpoints_path+"_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (latest_weights is not None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    model.load_weights(latest_weights)
    return model


def predict(model=None, inp=None, out_fname=None, checkpoints_path=None, out_fname2=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (inp is not None)
    assert((type(inp) is np.ndarray) or isinstance(inp, six.string_types)
           ), "Inupt should be the CV image or the input file name"

    inp = geotiff.imread(inp)

    if inp.shape[2] == 5:
        inp[:,:,4] = inp[:,:,4] - inp[:,:,4].min()
        inp[:,:,4] = inp[:,:,4]/(inp[:,:,4].max()+1)

    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)

    seg_img = cv2.resize(inp[:,:,0:3], (output_height, output_width), interpolation=cv2.INTER_AREA)
    colors = class_colors

    img_overlay = seg_img.copy()

    ch0 = img_overlay[:,:,0]
    ch1 = img_overlay[:,:,1]
    ch2 = img_overlay[:,:,2]
    NCLS = len(cfg_color.colors)
    for n in range(1,NCLS):
        idx = np.where(pr==n)
        color = np.array(cfg_color.colors[n])
        ch0[idx] = color[0]
        ch1[idx] = color[1]
        ch2[idx] = color[2]

    #seg_img = seg_img.astype(np.float32)
    seg_img = seg_img * 0.66 + img_overlay * 0.33
    #seg_img = seg_img.astype(np.uint8)

    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h), interpolation=cv2.INTER_NEAREST)

    cv2.imshow('Predict',seg_img/256)
    cv2.waitKey(10)

    if out_fname is not None:
        geotiff.imwrite(os.path.splitext(out_fname)[0]+'.tif', seg_img)


    if out_fname2 is not None:
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=0)
        cv2.imwrite(out_fname2, pr)
    return

def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None,
                     checkpoints_path=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    if inps is None and (inp_dir is not None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(
            os.path.join(inp_dir, "*.png")) + \
            glob.glob(os.path.join(inp_dir, "*.jpeg"))

    assert type(inps) is list

    all_prs = []

    for i, inp in enumerate(tqdm(inps)):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")

        pr = predict(model, inp, out_fname)
        all_prs.append(pr)

    return all_prs



def evaluate( model=None , inp_images=None , annotations=None,inp_images_dir=None ,annotations_dir=None , checkpoints_path=None ):
    pass


