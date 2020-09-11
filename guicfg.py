from collections import OrderedDict 
classes_dict = OrderedDict({'Squatter':1})
epochs = 200
predict_output_format = 'jpg'   # option include jpg bmp
MATLAB = False

def get(cls_name):
    if cls_name == "Vehicles":
        class cls_cfg:
            my_size = 576
            down_scale = 2
            bands = 5
    elif cls_name == "Trees":
        class cls_cfg:
            my_size = 576
            down_scale = 2
            bands = 5
    elif cls_name == 'BuildingTower':
        class cls_cfg:
            my_size = 576
            down_scale = 4
            bands = 5
    elif cls_name == 'Squatter':
        class cls_cfg:
            my_size = 512
            down_scale = 4
            bands = 5
    elif cls_name == 'Solar':
        class cls_cfg:
            my_size = 512
            down_scale = 2
            bands = 4
    elif cls_name == 'Farmland':
        class cls_cfg:
            my_size = 512
            down_scale = 4
            bands = 4
    elif cls_name == 'SquatterAndTower':
        class cls_cfg:
            my_size = 512
            down_scale = 4
            bands = 5
            cls_list = OrderedDict({'Squatter':1,'Tower':2})
    else:
        class cls_cfg:
            my_size = 512
            down_scale = 1
            bands = 4
    return cls_cfg
