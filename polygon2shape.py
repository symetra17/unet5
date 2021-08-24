import numpy as np
from numpy import uint8
import cv2
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import gdal
#import gdal_edit
import os
#from osgeo import gdal_array
gdal.PushErrorHandler('CPLQuietErrorHandler')


def draw_polygon(objects, rasterTemp, outShp):
            
    sourceRaster = gdal.Open(rasterTemp)
    band = sourceRaster.GetRasterBand(1)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    # If shapefile already exist, delete it
    if os.path.exists(outShp):
        driver.DeleteDataSource(outShp)
    outDatasource = driver.CreateDataSource(outShp)            
    srs = osr.SpatialReference()
    srs.ImportFromWkt( sourceRaster.GetProjectionRef() )
    outLayer = outDatasource.CreateLayer('predicted_object', srs, ogr.wkbPolygon)
    newField = ogr.FieldDefn('Class', ogr.OFTString)
    outLayer.CreateField(newField)
    outDatasource.Destroy()
    sourceRaster=None
    band=None
    ioShpFile = ogr.Open(outShp, update = 1)    
    lyr = ioShpFile.GetLayerByIndex(0)
    lyr.ResetReading()
    defn = lyr.GetLayerDefn()
    feat = ogr.Feature(defn)
    feat.SetField('id', 852)
    feat.SetField('Class', 'Grave')
    for obj in objects:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for pt in obj:
            ring.AddPoint(pt[0], pt[1])
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        str1 = poly.ExportToWkt()    # export to a string "POLYGON ((0 0 0,100 0 0,100 100 0,0 100 0,0 0 0))"
        geom = ogr.CreateGeometryFromWkt(str1)
        feat.SetGeometry(geom)      #feat.SetGeometry(geom)
        lyr.CreateFeature(feat)
    ioShpFile.Destroy()
    return


if __name__=='__main__':
    # note that this file should accompanied by a TFW file
    fileName = R"C:\Users\dva\Pictures\20200218SA1_B05_6NE14D.TIF"
    
    ds = gdal.Open(fileName)
    xoffset, px_w, _, yoffset, _, px_h = ds.GetGeoTransform()
    objects = []

    # first object
    point_list = []
    point_list.append((3375,801))
    point_list.append((3166,695))
    point_list.append((3253,531))
    point_list.append((3463,635))
    point_list.append((3375,801))

    point_list2 = []
    for pt in point_list:
        x = pt[0]
        y = pt[1]
        posX = px_w * x + xoffset
        posY = px_h * y + yoffset
        point_list2.append((posX, posY))
    objects.append(point_list2)

    # second  object
    point_list = []
    point_list.append((100,100))
    point_list.append((500,100))
    point_list.append((700,300))
    point_list.append((500,500))
    point_list.append((100,500))
    point_list.append((100,100))

    point_list2 = []
    for pt in point_list:
        x = pt[0]
        y = pt[1]
        posX = px_w * x + xoffset
        posY = px_h * y + yoffset
        point_list2.append((posX, posY))
    objects.append(point_list2)


    draw_polygon(objects, rasterTemp=str(fileName), outShp=str('shape_result'))

