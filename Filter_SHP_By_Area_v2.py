#!/usr/bin/env python
# coding: utf-8

# # Select the shapes with 'Area > 92.9 m2'

# In[1]:


from osgeo import ogr
import shapefile
import os
import io




# In[2]:


def write_log(file, info_txt):
    with io.open(file,"a",encoding="utf-8") as f:
        f.write(info_txt)
        f.write('\n')
        f.write('\n')


# #test\
# log_path = r"D:\Projects\Company\Temp\log2.log"\
# info_txt = "this is the 2nd line"\
# 
# write_log(log_path, info_txt)

# ### Add 'Area' into SHP

# In[3]:


# if the coloumn 'Area' is found, skip adding the coloumn
# create a new column 'Area' if it's not been found
def add_area(shp_path, log_path):
    source = ogr.Open(shp_path, update = True)
    layer = source.GetLayer()
    layer_defn = layer.GetLayerDefn()
    field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
    if 'Area' in field_names:
        info_txt ='Area Coloumn found in ' + shp_path
        
    else:
        # Add a new field
        new_field = ogr.FieldDefn('Area', ogr.OFTReal)
        new_field.SetWidth(22)
        new_field.SetPrecision(2) #added line to set precision
        layer.CreateField(new_field)
        info_txt ='Area Coloumn added in ' + shp_path
        
        for feature in layer:
            geom = feature.GetGeometryRef()
            area = geom.GetArea()
            # print(area)
            feature.SetField("Area", area)
            layer.SetFeature(feature)
            
    write_log(log_path, info_txt)
    # Close the Shapefile
    source = None


# #test\
# shp_path = r"D:\Projects\Company\Temp\predicted_object.shp"\
# add_area(shp_path)\

# ### Select features baesed on area

# In[4]:


# sf is the original shpfile, output_path is the output path of shpfile
# will select the features that Area > area_threshold
def select_shp_on_area(shp_path, output_shp_path, area_threshold):
    sf = shapefile.Reader(shp_path)
    w = shapefile.Writer(output_shp_path, shapeType = sf.shapeType)
    w.fields = list(sf.fields)
    
    for rec in sf.iterShapeRecords():
        ls = rec.record
        if ls.Area > area_threshold:
            w.record(*ls)
            w.shape(rec.shape)
    w.close()


# #test\
# shp_path = r"D:\Projects\Company\Temp\predicted_object.shp"\
# sf = shapefile.Reader(shp_path)\
# output_path = r"D:\Projects\Company\Temp\predicted_object5"\
# area_threshold = 13\
# select_shp_on_area(sf, output_path, area_threshold)

# ### Create Projection file for the new shp

# In[5]:


# create a projection file
def create_prj(shp_path, prj_path):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(shp_path)
    layer = dataset.GetLayer()
    spatialRef = layer.GetSpatialRef()
    
    file = open(prj_path, 'w')
    file.write(spatialRef.ExportToWkt())
    file.close()
    


# ### Batch Processing

# In[6]:


def batch_select_shp(current_path, output_path, area_threshold, Suffix):
    for root, dirs, files in os.walk(current_path, topdown=False):
        for name in files:
            if os.path.splitext(os.path.join(root, name))[1].lower() == ".shp":
                shpName = name.split('.')[0]
                shp_path = os.path.splitext(os.path.join(root, name))[0] + ".shp"
                prj_path = os.path.join(output_path, shpName) + Suffix + ".prj"
                log_path = os.path.join(output_path, 'logfile') + ".log"
                output_shp_path = os.path.join(output_path, shpName) + Suffix +".shp"
                
                sf = shapefile.Reader(shp_path)
                # when shp is polygon
                if sf.shapeType == 5:
                    # if shp is not empty
                    if len(sf.records()) > 0:
                        # if shp has projection information
                        if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".prj"):
                            info_txt = "The shp file -"+ name + " has projection information."
                            
                            create_prj(shp_path, prj_path)
                            
                         # if shp has no projection information  
                        else:
                            info_txt = "The shp file - " + name + " has no projection information."
                            
                        add_area(shp_path, log_path)
                        
                        select_shp_on_area(shp_path, output_shp_path, area_threshold)
                    
                else:
                    info_txt = "The shp file "+ shp_path + " is not polygon!"
                    
                write_log(log_path, info_txt)
                             


if __name__=='__main__':

    current_path = R'C:\Users\echo\Code\unet5\weights\Vehicles\VehiclesTS\20191128SA1_B05_6NE14A_shape'
    output_path = R'C:\Users\echo\Code\unet5\weights\Vehicles\VehiclesTS\20191128SA1_B05_6NE14A_shape\out'
    area_threshold = 92.9
    batch_select_shp(current_path, output_path, area_threshold,'kitty')


