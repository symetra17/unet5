from keras.models import *
from keras.layers import *
import keras.backend as K

from .config import IMAGE_ORDERING

def vanilla_encoder(bands, input_height=224,  input_width=224):

    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    img_input = Input(shape=(input_height, input_width, bands))

    x = img_input
    levels = []

    x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    x = (Conv2D(filter_size, (kernel, kernel),
                data_format=IMAGE_ORDERING, padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    levels.append(x)

    x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    x = (Conv2D(128, (kernel, kernel), data_format=IMAGE_ORDERING,
         padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    levels.append(x)

    for _ in range(3):
        x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
        x = (Conv2D(256, (kernel, kernel),
                    data_format=IMAGE_ORDERING, padding='valid'))(x)
        x = (BatchNormalization())(x)
        x = (Activation('relu'))(x)
        x = (MaxPooling2D((pool_size, pool_size),
             data_format=IMAGE_ORDERING))(x)
        levels.append(x)

    return img_input, levels
