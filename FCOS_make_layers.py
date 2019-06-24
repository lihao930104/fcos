import tensorflow as tf
from tensorflow.keras.layers import  Input, Add, Dense, \
    Activation,ZeroPadding2D, BatchNormalization, Flatten,Conv2D, \
     AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model ,Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.activations import relu
from tensorflow.keras.utils import  get_file ,plot_model
from tensorflow.keras.initializers import he_uniform
import tensorflow.keras.backend as K
import numpy as np
import math

def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(
        out_channel,kernel_size,stride=1,dilation=1
    ):
        model=Sequential()
        model.add(ZeroPadding2D(dilation*(kernal_size-1)//2)
        model.add(Conv2D(out_channel,kernel_size=kernel_size,stride=stride,\
            use_bias=False if use_gn else True,bias_initializer=Zeros if not use_gn, \
            kernel_initializer=he_uniform))
        if use_gn:
            model.add(group_norm(out_channel))
        if use_relu:
            model.add(relu())
        return model
    return make_conv
        

