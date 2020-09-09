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

    if isinstance(inp, six.string_types):
        inp = geotiff.imread(inp)
    
    assert inp.shape[2] == 5, "Image should be h,w,5 "
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

    img_overlay = np.zeros_like(seg_img)

    OUTPUT_BW = False
    if OUTPUT_BW:
        for n in range(pr.shape[0]):
            for m in range(pr.shape[1]):
                if pr[n,m] == 1:
                    seg_img[n, m, :] = np.array([255,255,255])
                else:
                    seg_img[n, m, :] = np.array([0,0,0])
            #else:
            #    # color overlay
            #    cls_id = pr[n,m]
            #    if cls_id > 0:
            #        seg_img[n, m, :] = seg_img[n, m, :]//2 + np.array(cfg.predict_color[cls_id])//2

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

    seg_img = seg_img//2 + img_overlay//2

    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h), interpolation=cv2.INTER_NEAREST)
    
    if out_fname is not None:
        #cv2.imwrite(out_fname, seg_img, [cv2.IMWRITE_JPEG_QUALITY, 97])
        cv2.imwrite(out_fname, seg_img)

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
    
    if model is None:
        assert (checkpoints_path is not None) , "Please provide the model or the checkpoints_path"
        model = model_from_checkpoint_path(checkpoints_path)
        
    if inp_images is None:
        assert (inp_images_dir is not None) , "Please privide inp_images or inp_images_dir"
        assert (annotations_dir is not None) , "Please privide inp_images or inp_images_dir"
        
        paths = get_pairs_from_paths(inp_images_dir , annotations_dir )
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])
        
    assert type(inp_images) is list
    assert type(annotations) is list
        
    tp = np.zeros( model.n_classes  )
    fp = np.zeros( model.n_classes  )
    fn = np.zeros( model.n_classes  )
    n_pixels = np.zeros( model.n_classes  )
    
    for inp , ann   in tqdm( zip( inp_images , annotations )):
        pr = predict(model , inp )
        gt = get_segmentation_array( ann , model.n_classes ,  model.output_width , model.output_height , no_reshape=True  )
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()
                
        for cl_i in range(model.n_classes ):
            
            tp[ cl_i ] += np.sum( (pr == cl_i) * (gt == cl_i) )
            fp[ cl_i ] += np.sum( (pr == cl_i) * ((gt != cl_i)) )
            fn[ cl_i ] += np.sum( (pr != cl_i) * ((gt == cl_i)) )
            n_pixels[ cl_i ] += np.sum( gt == cl_i  )
            
    cl_wise_score = tp / ( tp + fp + fn + 0.000000000001 )
    n_pixels_norm = n_pixels /  np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)
    return {"frequency_weighted_IU":frequency_weighted_IU , "mean_IU":mean_IU , "class_wise_IU":cl_wise_score }


