from collections import OrderedDict 
classes_dict = OrderedDict({'Squatter':1})
epochs = 200
predict_output_format = 'jpg'   # option include jpg bmp
show_training_page = True

augm_rotation = True
augm_angle_range = [-30, 30]

cls_list = ['Farmland','Trees','Vehicles','Squatter','Solar', 'Algae']

NUM_OF_PRED_PROC = 1

def get(cls_name):
    if cls_name == "Vehicles":
        class cls_cfg:
            my_size = 512
            down_scale = 1
            bands = 4
            cls_sub_list = OrderedDict({'Vehicles':1})
            discard_empty = 0.8   # discard some training images without object, 0.8 for dropping 80% 
    elif cls_name == "Trees":
        class cls_cfg:
            my_size = 576
            down_scale = 2
            bands = 5
            cls_sub_list = OrderedDict({'Trees':1})
            discard_empty = 0.8   # discard some training images without object, 0.8 for dropping 80% 
    elif cls_name == 'Squatter':
        class cls_cfg:
            my_size = 512
            down_scale = 4
            bands = 5
            cls_sub_list = OrderedDict({'Squatter':1,'BuildingTower':2})
            discard_empty = 0.1   # discard some training images without object, 0.8 for dropping 80% 
    elif cls_name == 'Solar':
        class cls_cfg:
            my_size = 512
            down_scale = 1
            bands = 4
            cls_sub_list = OrderedDict({'Solar':1})
            discard_empty = 0.98   # discard some training images without object, 0.8 for dropping 80% 
    elif cls_name == 'Farmland':
        class cls_cfg:
            my_size = 512
            down_scale = 4
            bands = 4
            cls_sub_list = OrderedDict({'Farmland':1})
            discard_empty = 0.8   # discard some training images without object, 0.8 for dropping 80% 
    elif cls_name == 'Algae':
        class cls_cfg:
            my_size = 224
            down_scale = 1
            bands = 3
            cls_sub_list = OrderedDict({'long':1, 'medium':2, 'short':3})
            discard_empty = 0.01   # discard some training images without object, 0.8 for dropping 80% 
    return cls_cfg
