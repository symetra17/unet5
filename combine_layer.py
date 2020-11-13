import numpy as np
import os
import geotiff
import cv2
from glob import glob

folder1 = R'C:\Users\echo\Pictures\lands_inference_example\DOM'
folder2 = R'C:\Users\echo\Pictures\lands_inference_example\DSM'

dom_files = glob(os.path.join(folder1, '*.tif'))
dom_files.sort()

dsm_files = glob(os.path.join(folder2, '*.tif'))
dsm_files.sort()

for n, f in enumerate(dom_files):
    print(dom_files[n])
    print(dsm_files[n])
    print('')

    f1 = dom_files[n]
    f2 = dsm_files[n]

    a1 = geotiff.imread(f1)
    a2 = geotiff.imread(f2)
    assert len(a1.shape) == 3
    assert len(a2.shape) == 2
    a2 = np.expand_dims(a2, axis=2)
    a3 = np.concatenate((a1,a2), axis=2)
    print(a3.shape)
    a3 = cv2.resize(a3, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    print(a3.shape)

    geotiff.imwrite('out.tif',a3)

    quit()

