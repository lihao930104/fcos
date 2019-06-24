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
from .FCOS_resnet import Resnet
from .FCOS_fpn import Fpn,LastLevelP6P7
from .FCOS_make_layers import conv_with_kaiming_uniform

def build_resnet_fpn_backbone(inputs,cfg):
    resout=Resnet(inputs,cfg)
    in_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels=cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpnout=Fpn(
        resout
        in_channels_list=[
            in_channels,
            in_channels*2,
            in_channels*4,
            in_channels*8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU)
        cfg,
        top_block="fpn"
    )
    return Model(inputs,fpnout)

def build_resnet_fpn_p3p7_backbone(inputs,cfg):
    resout=Resnet(inputs,cfg)
    in_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels=cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7=in_channels*8 if  cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpnout=Fpn(
        resout
        in_channels_list=[
            0,
            in_channels*2,
            in_channels*4,
            in_channels*8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU)
        cfg,
        top_block="fpnp6p7"
    )
    return Model(inputs,fpnout)