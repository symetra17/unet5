from collections import OrderedDict 
classes_dict = OrderedDict({'Squatter':1})
epochs = 200
predict_output_format = 'jpg'   # option include jpg bmp
MATLAB = False

cls_list = ['Farmland','Trees','Vehicles','Squatter','Solar']

def get(cls_name):
    if cls_name == "Vehicles":
        class cls_cfg:
            my_size = 576
            down_scale = 2
            bands = 5
            cls_sub_list = OrderedDict({'Vehicles':1})
    elif cls_name == "Trees":
        class cls_cfg:
            my_size = 576
            down_scale = 2
            bands = 5
            cls_sub_list = OrderedDict({'Trees':1})
    elif cls_name == 'Squatter':
        class cls_cfg:
            my_size = 512
            down_scale = 4
            bands = 5
            cls_sub_list = OrderedDict({'Squatter':1,'BuildingTower':2})
    elif cls_name == 'Solar':
        class cls_cfg:
            my_size = 512
            down_scale = 1
            bands = 4
            cls_sub_list = OrderedDict({'Solar':1})
    elif cls_name == 'Farmland':
        class cls_cfg:
            my_size = 512
            down_scale = 4
            bands = 4
            cls_sub_list = OrderedDict({'Farmland':1})
    return cls_cfg
