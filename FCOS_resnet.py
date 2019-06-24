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


_STEM_MODULES = dict({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
    "StemWithGN": StemWithGN,
})

_TRANSFORMATION_MODULES = dict({
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm,
    "BottleneckWithGN": BottleneckWithGN,
})



_STAGE_SPECS = dict({
    "R-50-C4": ResNet50StagesTo4,
    "R-50-C5": ResNet50StagesTo5,
    "R-101-C4": ResNet101StagesTo4,
    "R-101-C5": ResNet101StagesTo5,
    "R-50-FPN": ResNet50FPNStagesTo5,
    "R-50-FPN-RETINANET": ResNet50FPNStagesTo5,
    "R-101-FPN": ResNet101FPNStagesTo5,
    "R-101-FPN-RETINANET": ResNet101FPNStagesTo5,
    "R-152-FPN": ResNet152FPNStagesTo5,
})

StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Number of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

ResNet50StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)
# ResNet-50 up to stage 4 (excludes stage 5)
ResNet50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)
# ResNet-101 (including all stages)
ResNet101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True))
)
# ResNet-101 up to stage 4 (excludes stage 5)
ResNet101StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, True))
)
# ResNet-50-FPN (including all stages)
ResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)
# ResNet-101-FPN (including all stages)
ResNet101FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True))
)
# ResNet-152-FPN (including all stages)
ResNet152FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True))

def Resnet(inputs,cfg):
     stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
     transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]
     stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
     # Constuct the specified ResNet stages
     num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
     width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
     in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
     stage_bottleneck_channels = num_groups * width_per_group
     stage_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
     stages = []
     return_features ={}
     stages_models=dict{}
     for stage_spec in stage_specs:
          name="layer"+str(stage_spec)
          stage2_relative_factor=2**(stage_spec.index-1)
          stabottleneck_channels =stage_bottleneck_channels*stage_relative_factor
          out_channels=stage_out_channels*stage_relative_factor
          stage =_make_stage(
               inputs,
               transformation_module,
               in_channels,
               bottleneckchannels,
               out_channels,
               stage_spec.block_count,
               cfg.MODEL.RESNETS.STRIDE_IN_1X1,
               fist_stride=int(stage_spec.index>1)+1
          )
          in_channels=out_channels
        
          stages_models[name]=stage
          stages.append(name)
          return_features[name]=stage_spec.return_features
          """def _freeze_backbone(freeze_at):
               if freeze_at<0:
                    return
               for stage_index in range(freeze_at):
                    if stage_index==0:
                         m=stem_module
                    else：
                         pass

          freeze=_freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)"""


     outputs=[]
     # Construct the stem module
     out = stem_module(inputs,cfg)
     for stage_name in stages:
          out=stages_models[stage_name](out)
          if return_features[stage_name]:
               outputs.append(out)
     returns outputs



                          
def _make_stage(
     inputs,
     transformation_module,
     in_channels,
     bottleneckchannels,
     out_channels,
     block_count,
     stride_1X1,
     fist_stride,
     dilation
     ): 
      
     sride=fist_stride
     for _ in range(block_count):
          out=transformation_module(
                    inputs,
                    in_channels,
                    bottleneckchannels,
                    out_channels,
                    stride_1X1,
                    stride,
                    dilation=dilation
               )
          inputs=out     
          stride=1
          in_channels=out_channels
     return out


def Basestem(inputs,cfg):
    out_channels =cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    x = ZeroPadding2D(padding=(3,3))(inputs)
    x = Conv2D(out_channels,kernal_size=(7,7),stride=2,padding='valid',\
         use_bias =False,kernal_initializer='he_uniform')(x)#he_uniform无法定义变量a
    x = norm_func(out_channels)(x)#定义NORM_func
    x = Relu(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    out = MaxPooling2D(kernal_size=(3,3),stride=2,padding='valid')(x)
    return out

def BottleneckwithGN(
     inputs,
     in_channels,
     bottleneckchannels,
     out_channels,
     stride_in_1X1=True,
     stride=1
     dilation=1
     ):

     downsample=None
     if in_channels!=out_channels:
          down_stride=stride if dilation==1 else 1
          downsample=([
               Conv2D(out_channels,kernel_size=(1,1),stride=down_stride,bias=False,\
                    kernel_regularizer=he_uniform),#he_uniform无法定义变量a
               norm_func(out_channels),
          ])

     if dilation>1:
          stride=1
     stride_1X1,stride_3x3 =(stride,1) if stride_in_1X1 else (1,stride)

     identity=inputs

     x=Conv2D(bottleneckchannels,kernel_size=1,stride=stride_1X1,use_bias=False,dilation=dilation,\
          kernel_initializer=he_uniform)(inputs)#he_uniform无法定义变量a
     x=norm_func(bottleneckchannels)(x)
     x=relu(x) 

     x=ZeroPadding2D(padding=dilation)(x)
     x=Conv2D(bottleneckchannels,kernel_size=3,stride=stride_3x3,use_bias=False,\
          kernel_initializer=he_uniform)(x) #he_uniform无法定义变量a
     x=norm_func(bottleneckchannels)(x)
     x=relu(x)

     x=Conv2D(out_channels,kernel_size=1,use_bias=False,stride=1,\
          kernel_initializer=he_uniform)(x) #he_uniform无法定义变量a
     out=norm_func(out_channels)(x)

     if downsample is not None:
          identity =downsample(inputs)

     out+=identity
     out=relu(out)

     return out


def norm_func():
     pass




     


     

     
   
