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
    elif cls_name == "Trees":
        class cls_cfg:
            my_size = 576
            down_scale = 2
    elif cls_name == 'BuildingTower':
        class cls_cfg:
            my_size = 576
            down_scale = 4
    elif cls_name == 'Squatter':
        class cls_cfg:
            my_size = 512
            down_scale = 4
    else:
        class cls_cfg:
            my_size = 640
            down_scale = 4
    return cls_cfg
