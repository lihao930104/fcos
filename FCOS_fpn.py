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

def Fpn(inputs,in_channels_list,out_channels,conv_block,cfg,top_blocks="fpn"):
  
    #inner_block=[]
    #layer_block=[]
    models_inners=[]
    models_layers=[]
    results=[]
    for in_channels in in_channels_list:
        #inner_block="fpn_inner{}".format(idx)
        #layer_block="fpn_layer{}".format(idx)
        if in_channels==0:
            continue
        inner_block_module=conv_block(out_channels,1)
        layer_block_module=conv_block(out_channels,3,1)
        models_inners.append(inner_block_module)
        models_layers.append(layer_block_module)
    last_inner=models_inners[-1](inputs[-1])
    last_layers=models_layers[-1](last_inner)
    results.append(last_layers)
    for feature,inner_block,layer_block in zip(
        inputs[:-1][::-1],models_inners[:-1][::-1],models_layers[:-1][::-1]
    ):
        if not inner_block:
            continue
        inner_top_down=UpSampling2D(size=(2,2),interpolation="nearest")(last_inner)
        inner_lateral=inner_block(feature)
        last_inner=inner_lateral+inner_top_down
        last_layers=layer_block(last_inner)

        results.insert(0,last_layers)
    if top_block=="fpn_p6p7":
        last_results=LastLevelP6P7(inputs[-1],results[-1],out_channels,cfg)
        results.extend(last_results)
    if top_block=="fpn":
        last_results=MaxPooling2D(1,2)(results[-1])
        results.extend(last_results)
    
    return results

def LastLevelP6P7(c5,p5,out_channels,cfg):
    inputs=c5 if cfg.MODEL.RETINANET.USE_C5 else p5
    p6=ZeroPadding2D(padding=(1,1))(inputs)
    p6=Conv2D(out_channels,3,2,padding='valid',use_bias=False,kernel_initializer=he_uniform)(p6)
    p6=relu(p6)
    p7=ZeroPadding2D(padding=(1,1))(p6)
    p7=Conv2D(out_channels,3,2,padding='valid',use_bias=False,kernel_initializer=he_uniform)(p7)
    
    return[p6,p7]



    


       




