# Select the shapes with 'Area > 92.9 m2'
from osgeo import ogr
import shapefile
import os
import io


def write_log(file, info_txt):
    with io.open(file,"a",encoding="utf-8") as f:
        f.write(info_txt)
        f.write('\n')
        f.write('\n')

# if the coloumn 'Area' is found, skip adding the coloumn
# create a new column 'Area' if it's not been found
def add_area(shp_path,category):
    source = ogr.Open(shp_path, update = True)
    layer = source.GetLayer()
    layer_defn = layer.GetLayerDefn()
    field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
    if 'Area' in field_names:
        info_txt ='Area Coloumn found in ' + shp_path
    else:
        # Add 2 new fields
        new_field = ogr.FieldDefn('Category', ogr.OFTString)
        new_field.SetWidth(22)
        layer.CreateField(new_field)
            
        new_field = ogr.FieldDefn('Area', ogr.OFTReal)
        new_field.SetWidth(22)
        new_field.SetPrecision(2)
        layer.CreateField(new_field)
        info_txt ='Area Coloumn added in ' + shp_path        
        for feature in layer:
            geom = feature.GetGeometryRef()
            area = geom.GetArea()
            feature.SetField("Area", area)
            feature.SetField("Category", category)
            layer.SetFeature(feature)

    # Close the Shapefile
    source = None


# #test\
# shp_path = r"D:\Projects\Company\Temp\predicted_object.shp"\
# add_area(shp_path)\

# ### Select features baesed on area

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


# create a projection file
def create_prj(shp_path, prj_path):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(shp_path)
    layer = dataset.GetLayer()
    spatialRef = layer.GetSpatialRef()
    if spatialRef is not None:
        file = open(prj_path, 'w')
        file.write(spatialRef.ExportToWkt())
        file.close()
    


def batch_select_shp(current_path, output_path, area_threshold, Suffix):
    for root, dirs, files in os.walk(current_path, topdown=False):
        for name in files:
            if os.path.splitext(os.path.join(root, name))[1].lower() == ".shp":
                shpName = name.split('.')[0]
                shp_path = os.path.splitext(os.path.join(root, name))[0] + ".shp"
                prj_path = os.path.join(output_path, shpName) + Suffix + ".prj"
                
                output_shp_path = os.path.join(output_path, shpName) + Suffix +".shp"
                
                sf = shapefile.Reader(shp_path)
                # when shp is polygon
                if sf.shapeType == 5:
                    # if shp is not empty
                    if len(sf.records()) > 0:
                        # if shp has projection information
                        if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".prj"):
                            create_prj(shp_path, prj_path)                            
                         # if shp has no projection information  
                        else:
                            pass
                        category = 'cat1'
                        add_area(shp_path,category)
                        select_shp_on_area(shp_path, output_shp_path, area_threshold)
                    

def add_area_single(fname, area_threshold, output_shp_path, category):
    shpName = os.path.splitext(fname)[0]
    shp_path = fname
    prj_path = shpName + ".prj"
    #output_shp_path = shpName + Suffix +".shp"
    sf = shapefile.Reader(shp_path)
    # when shp is polygon
    if sf.shapeType == 5:
        # if shp is not empty
        if len(sf.records()) > 0:
            # if shp has projection information
            if os.path.exists(prj_path):
                create_prj(shp_path, prj_path)
             # if shp has no projection information  
            else:
                pass
            add_area(shp_path,category)
            select_shp_on_area(shp_path, output_shp_path, area_threshold)        


if __name__=='__main__':
    fname = R"C:\Users\echo\Code\unet5\weights\Solar\Newfolder\ssss\predicted_object.shp"
    add_area_single(fname,1.0,'_echo','Solar')
    quit()

    current_path = R'C:\Users\echo\Code\unet5\weights\Vehicles\VehiclesTS\20191128SA1_B05_6NE14A_shape'
    output_path = R'C:\Users\echo\Code\unet5\weights\Vehicles\VehiclesTS\20191128SA1_B05_6NE14A_shape\out'
    area_threshold = 92.9
    batch_select_shp(current_path, output_path, area_threshold,'kitty')


