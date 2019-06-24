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
from .FCOS_head import fcoshead
from .FCOS_loss import fcos_loss_eval
from .FCOS_postprocess import fcospostprocessor

def FCOSmodule(images,features,cfg,inchannels,targets=None):

    box_cls,box_regression,centerness = FCOSHead((features,cfg,inchannels)
    locations = compute_locations(features,cfg)
    losses = fcos_loss_eval(cfg,locations,box_cls,box_regression,centerness,targets)
    boxes = fcospostprocessor(locations,box_cls,box_regression,centerness,images.image_sizes)

    return None





def compute_locations(inputs,cfg):
    locations=[]
    for l,feature in enumerate(inputs):
        h,w=feature.shape()[-2:]
        locations_per_level=compute_locations_per_level(
            h,w, cfg.MODEL.FCOS.FPN_STRIDES[l]
        )
        locations.append(locations_per_level)
    return locations

def compute_locations_per_level(h,w,stride):
    shifts_x=K.arange(start=0,stop=w*stride,step=stride,dtype='float32')
    shifts_y=K.arange(start=0,stop=h*stride,step=stride,dtype='float32')
    gird_x=K.tile(K.reshape(shifts_x,[1,-1,1]),[h,1,1])
    grid_y=K.tile(K.reshape(shifts_y,[-1,1,1)),[1,w,1])
    grid=K.concatenate([grid_x,grid_y])
    location=gird+stride//2
    location=location.reshape(-1,2)
    return location
