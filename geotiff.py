from PIL import Image
import numpy as np
from numpy import uint8
import cv2
from cv2 import imshow, waitKey
#from gdalconst import *
import gdalconst
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import gdal
import gdal_edit
import os
from osgeo import gdal_array

Image.MAX_IMAGE_PIXELS = None

def GetGeoInfo(FileName):
    # Example of SpatialReference
    '''
    PROJCS["Hong Kong 1980 Grid System",
    GEOGCS["Hong Kong 1980",
        DATUM["Hong_Kong_1980",
            SPHEROID["International 1924",6378388,297,
                AUTHORITY["EPSG","7022"]],
            AUTHORITY["EPSG","6611"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4611"]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",22.3121333333333],
    PARAMETER["central_meridian",114.178555555556],
    PARAMETER["scale_factor",1],
    PARAMETER["false_easting",836694.05],
    PARAMETER["false_northing",819069.8],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Northing",NORTH],
    AXIS["Easting",EAST],
    AUTHORITY["EPSG","2326"]]
    '''
    SourceDS = gdal.Open(FileName, gdalconst.GA_ReadOnly)
    Projection = osr.SpatialReference()
    Projection.ImportFromWkt(SourceDS.GetProjectionRef())
    return Projection

def CreateGeoTiff(outname, Array, GeoT, Projection):
    xsize = Array.shape[1]
    ysize =  Array.shape[0]
    DataType = gdal.GDT_Byte
    # Set up the dataset
    driver = gdal.GetDriverByName("GTiff")
    DataSet = driver.Create( outname, xsize, ysize, Array.shape[2], DataType )
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection( Projection.ExportToWkt() )
    # Write the array
    for n in range(Array.shape[2]):
        DataSet.GetRasterBand(n+1).WriteArray( Array[:,:,n] )

def read(infile):
    ds = gdal.Open(infile, gdal.GA_ReadOnly)  # type(ds) = osgeo.gdal.Dataset
    gt = ds.GetGeoTransform()
    rb = ds.GetRasterBand(1)
    img_array = rb.ReadAsArray()    
    from osgeo import gdal_array
    npa = gdal_array.LoadFile(infile)
    return gt, npa

def generate_tif_alpha(np4ch, outname, geotiff_ref_in):
    # we copy the geo info from geotiff_ref, and put them in the new geotiff
    targetProj = GetGeoInfo(geotiff_ref_in)
    label = 'Hong Kong 1980'
    if label in str(targetProj):
        print('Projection:', label)
        gt, pix = read(geotiff_ref_in)
        new_fname = CreateGeoTiff(outname, np4ch, gt, targetProj)
        gdal_edit.gdal_edit(['','-colorinterp_1', 'red', '-colorinterp_2', 'green', 
                '-colorinterp_3', 'blue', '-colorinterp_4', 'alpha', outname])

def is_geotif(fname):
    targetProj = GetGeoInfo(fname)
    label = 'Hong Kong 1980'
    if label in str(targetProj):
        return True
    else:
        return False

def imread(fname):  # read like opencv
    from osgeo import gdal_array
    npa = gdal_array.LoadFile(fname)
    npa = np.moveaxis(npa, 0, 2)
    npa = npa.astype(np.float32)
    return npa

def imwrite(fname,npa):
    npa = np.moveaxis(npa, 2, 0)  # channel last to channel first
    gdal_array.SaveArray(npa, fname, format="GTiff")

def polygonize(rasterTemp, outShp):
            
            sourceRaster = gdal.Open(rasterTemp)
            band = sourceRaster.GetRasterBand(1)
            driver = ogr.GetDriverByName("ESRI Shapefile")
            # If shapefile already exist, delete it
            if os.path.exists(outShp):
                driver.DeleteDataSource(outShp)
            outDatasource = driver.CreateDataSource(outShp)            
            # get proj from raster            
            srs = osr.SpatialReference()
            srs.ImportFromWkt( sourceRaster.GetProjectionRef() )
            # create layer with proj

            outLayer = outDatasource.CreateLayer('predicted_object', srs)
            # Add class column (1,2...) to shapefile
      
            newField = ogr.FieldDefn('Class', ogr.OFTInteger)
            outLayer.CreateField(newField)
            
            gdal.Polygonize(band, None, outLayer, 0,[],callback=None)
            outDatasource.Destroy()
            sourceRaster=None
            band=None
            ioShpFile = ogr.Open(outShp, update = 1)
            lyr = ioShpFile.GetLayerByIndex(0)
            lyr.ResetReading()

            for n, i in enumerate(lyr):
                lyr.SetFeature(i)
                # if area is less than inMinSize or if it isn't forest, remove polygon 
                if i.GetField('Class')!=255:
                    lyr.DeleteFeature(i.GetFID())
            ioShpFile.Destroy()
            return

def adjust_gamma_16bit(image, gamma=1.4, post_level=0.014):    
    image = image.astype(np.float)
    invGamma = 1.0 / gamma
    img2 = 65535.0 * np.power((image/65535.0), invGamma)
    img2 = img2 * post_level
    img2[img2>255] = 255
    return img2.astype(np.uint16)

def gdal_imread_as_uint8(fname):
    ds = gdal.Open(fname, gdal.GA_ReadOnly)
    rb = ds.GetRasterBand(1)
    img = rb.ReadAsArray()
    h = img.shape[0]
    w = img.shape[1]
    img_tri_8 = np.zeros((h,w,3), dtype=np.uint8)
    img = adjust_gamma_16bit(img)
    img_tri_8[:,:,2] = img.astype(np.uint8)
    rb = ds.GetRasterBand(2)
    img = rb.ReadAsArray()
    img = adjust_gamma_16bit(img)
    img_tri_8[:,:,1] = img.astype(np.uint8)
    rb = ds.GetRasterBand(3)
    img = rb.ReadAsArray()
    img = adjust_gamma_16bit(img)    
    img_tri_8[:,:,0] = img.astype(np.uint8)
    return img_tri_8

if __name__=='__main__':
    fname = R"C:\Users\echo\Code\unet5\weights\5stack\20180313SA1_B05_6NW14C.tif"
    img = gdal_array.LoadFile(fname)
    img = img[4,:,:]
    img = img.astype(np.uint8)
    gdal_array.SaveArray(img, R"C:\Users\echo\Code\unet5\weights\5stack\20180313SA1_B05_6NW14C_3ch.tif", 
                    format="bmp")
    quit()

    geotiff_ref_in = R"C:\Users\dva\Pictures\20191128SA1_B05_6NE19C.TIF"
    
    img = imread(R"C:\Users\echo\Pictures\20180313SA1_B05_6NW14C.tif")
    cv2.imwrite('out.bmp', img[:,:,4])

    #outname = 'out.tif'
    #np4ch = np.zeros((12000,15000,4),dtype=uint8)
    #np4ch[:,:,3] = 255
    #generate_tif_alpha(np4ch, outname, geotiff_ref_in)
    
    inp = R"C:\Users\dva\Pictures\20191128SA1_B05_6NE19C_result2.tif"
    polygonize(inp, R"c:\users\dva\pictures\my_shape")

