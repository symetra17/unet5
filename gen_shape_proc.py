import numpy as np
from pathlib import Path
import geotiff
import os
import shp_filter
import multiprocessing as mp

def replace_ext(inp, new_ext):
    result = os.path.splitext(inp)[0] + new_ext
    return result

def gen_shape_proc(src_fname, img, class_name, output_folder):
        shape1 = geotiff.get_img_h_w(src_fname)
        src_h = shape1[0]
        src_w = shape1[1]

        np4ch = np.zeros((src_h, src_w, 4), dtype=np.uint8)
        cp = img[0:src_h, 0:src_w, 0].astype(np.uint8).copy()
        np4ch[:,:,3] = cp   # fill alpha channel with detection result
        np4ch[:,:,0] = cp   # fill 1st channel with detection result
        out_path = Path(replace_ext(src_fname, '_result_geo.tif'))
        out_path = Path(output_folder, out_path.name)
        geotiff.generate_tif_alpha(np4ch, str(out_path), src_fname)

        result_mask_path = out_path
        out_shp_file = Path(os.path.splitext(src_fname)[0] + '_shape')
        out_shp_file = Path(output_folder, Path(out_shp_file.name))

        geotiff.polygonize(rasterTemp=str(result_mask_path), outShp=str(out_shp_file))
        shp_filter.add_area_single(str(out_shp_file/'predicted_object.shp'), 
                10, out_shp_file/'filtered'/'predicted_object.shp', class_name)
