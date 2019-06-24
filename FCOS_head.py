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


def fcoshead(inputs,cfg,inchannels):
    num_classes= num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
    logits = []
    bbox_reg = []
    center = []
    scales=[]
    for i in range(5):
        scales.append(scale)
  
    for l,feature in enumerate(inputs):
        cls_tower,box_tower=feature
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            cls_tower=Conv2D(inchannels,kernel_size=3,stride=1,padding="same",\
                kernel_initializer=Randomnormal(stddev=0.01),use_bias=True,bias_initializer=Zeros)(cls_tower)
            cls_tower=group_norm(32,inchannels)(cls_tower)
            cls_tower=relu(cls_tower)
            
            box_tower=Conv2D(inchannels,kernel_size=3,stride=1,padding="same",\
                kernel_initializer=Randomnormal(stddev=0.01),use_bias=True,bias_initializer=Zeros)(box_tower)
            box_tower=group_norm(32,inchannels)(box_tower)
            box_tower=relu(box_tower)
        

        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        cls_logits=Conv2D(num_classes,kernel_size=3,stride=1,padding="same",\
            kernel_initializer=Randomnormal(stddev=0.01),use_bias=True,\
            bias_initializer=Constant(bias_value))(cls_tower)

        bbox_pred=Conv2D(4,kernel_size=3,stride=1,padding="same",\
            kernel_initializer=Randomnormal(stddev=0.01),use_bias=True,bias_initializer=Zeros)(box_tower)

        centerness=Conv2D(1,kernel_size=3,stride=1,padding="same",\
            kernel_initializer=Randomnormal(stddev=0.01),use_bias=True,bias_initializer=Zeros)(cls_tower)

        logits.append(cls_logits)
        bbox_reg.append(exp(scales[l](pbbox_pred)))
        center.append(centerness)

    return logits,bbox_reg,center

class scale(Layer):
    def __init__(self):
        
        super(scale, self).__init__()
 
    def build(self,):
        
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1), 
                                      initializers=Ones,
                                      trainable=True)     
        super(scale, self).build()  
 
    def call(self, x):
        return x * self.kernel


                            

