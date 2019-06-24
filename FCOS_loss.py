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


INF = 100000000



def fcos_loss_eval(cfg,locations,box_cls,box_regression,centerness,targets):
    class_loss=Sigmoidfocalloss(
        cfg.MODEL.FCOS.LOSS_GAMMA,
        cfg.MODEL.FCOS.LOSS_ALPHA)
    reg_loss=IOUloss()
    cen_loss=categorical_crossentropy()
    N=box_cls[0].shape[0]
    num_classes=box_cls[0].size[3]
    labels,reg_targets=prepare_targets(locations,targets)

    box_cls_flatten = []
    box_regression_flatten = []
    centerness_flatten = []
    labels_flatten = []
    reg_targets_flatten = []

    for l in range(len(labels)):
        box_cls_flatten.append(box_cls[l].reshape(-1,num_classes))#所有图片在一张特征图上
        box_regression_flatten.append(box_regression[l].reshape(-1,4))
        labels_flatten.append(labels[l].reshape(-1))
        reg_targets_flatten.append(reg_targets[l].reshape(-1,4))
        centerness_flatten.append(centerness[l].reshape(-1))
    
    box_cls_flatten=K.concatenate(box_cls_flatten,axis=0)#所有图片在所有特征图上
    box_regression_flatten=K.concatenate(box_regression_flatten.axis=0)
    centerness_flatten=K.concatenate(centerness_flatten,axis=0)
    labels_flatten=K.concatenate(labels_flatten,axis=0)
    reg_targets_flatten=K.concatenate(reg_targets_flatten,axis=0)
    mask=np.nonzero(labels_flatten.eval())
    num_point_label_nonzero=mask[0].shape[0]
    cls_loss=class_loss(
        box_cls_flatten,
        K.cast(labels_flatten,dtype='int32')
    )/(num_point_label_nonzero+N)

    box_regression_flatten = K.gather(box_regression_flatten,mask[0])
    reg_targets_flatten = K.gather(reg_targets_flatten,mask[0])
    centerness_flatten = K.gather(centerness_flatten,mask[0])

    if num_point_label_nonzero>0：
        centerness_targets=compute_centerness_targets(reg_targets_flatten)
        reg_loss = reg_loss(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
        )
        centerness_loss = cen_loss(
                centerness_flatten,
                centerness_targets
        )
    else:

        reg_loss = box_regression_flatten.sum()
        centerness_loss = centerness_flatten.sum()

    return cls_loss, reg_loss, centerness_loss

def compute_centerness_targets(reg_targets_flatten):
    left_right=K.concatenate(
        reg_targets_flatten[:,0].reshape(-1,1),reg_targets_flatten[:,2].reshape(-1,1),axis=-1
    ) 
    top_bottom=K.concatenate(
        reg_targets_flatten[:,1].reshape(-1,1),reg_targets_flatten[:,3].reshape(-1,1),axis=-1
    )
    centerness=(
        K.min(left_right,axis=-1)/K.max(left_right,axis=-1)*\
            K.min(top_bottom,axis=-1)/K.max(top_bottom,axis=-1)
    )
    centerness=K.sqrt(centerness)
    return centerness




def prepare_targets(points,targets):
    object_size_of_interest = [
        [-1,64],
        [64,128]
        [128,256]
        [256,512]
        [512,INF],
    ]
    expanded_object_sizes_of_interest=[]
    for l,points_per_level in enumerate(points):
        ob_size_of_per_level=K.variable(object_size_of_interest[l],dtype=K.dtype(points))
        expanded_object_sizes_of_interest.append(
            K.tile(K.reshape(ob_size_of_per_level,[1,2]),[len(points_per_level,1)])#(一层特征图的像素数，2)
        )
    expanded_object_sizes_of_interest=K.concatenate(expanded_object_sizes_of_interest,axis=0)
    #(所有特征图的像素总和,2)
    num_point_per_level=[len(points_per_level) for points_per_level in points]
    points_all_level=K.concatenate(points,axis=0)#(所有特征图的像素总和,2)
    labels, reg_targets = compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
    )
    for i in range(len(labels):
        labels[i]=tf.split(labels[i],num_point_per_level,axis=0)
        reg_targets[i]=tf.split(reg_targets[i],num_point_per_level,axis=0)
    
    labels_level_first = []
    reg_targets_level_first = []
    for level in range(len(points)):
        labels_level_first.append(
            K.concatenate([labels_per_image[level] for labels_per_image in labels],axis=0)
        )
        reg_targets_level_first.append(
            K.concatenate([reg_tagets_per_image[level] for reg_tagets_per_image in reg_targets],axis=0)
        )

    return labels_level_first, reg_targets_level_first




def compute_targets_for_locations(locations,targets,expanded_object_sizes_of_interest):

    labels=[]
    reg_targets=[]
    xs,ys=locations[:,0],locations[:,1]
    for im_i in range(len(targets))：
        targets_per_im=targets[im_i]
        bboxes=targets_per_im['detections']#(m,4)
        labels_per_im=targets_per_im['labels']#(m,)
        area=targets_per_im['aera']#(m,)

        l=xs[:,None]-bboxes[:,0][None]#(n,1)-(1,m)=(n,m) n为所有特征图的像素总和,m为标签框的个数
        t=ys[:,None]-bboxes[:,1][None]#(n,1)-(1,m)=(n,m)
        r=bboxes[:,2][None]-xs[:,None]#(n,1)-(1,m)=(n,m)
        b=bboxes[:,3][None]-ys[:,None]#(n,1)-(1,m)=(n,m)
        reg_targets_per_im=tf.stack([l,t,r,b],axis=2)#(n,m,4)

        is_in_boxes=K.min(reg_targets_per_im,axis=2)>0#(n,m)
        max_reg_targets_per_im=K.max(reg_targets_per_im,axis=2)#(n,m)
        is_called_in_the_level=\ 
            (max_reg_targets_per_im>=expanded_object_sizes_of_interest[:,[0]])&、\
            (max_reg_targets_per_im<=expanded_object_sizes_of_interest[:,[1]])#(n,m)
        locations_TO_GT_AREA=K.tile(area[None],[len(locations),1])#(n,m)
        locations_TO_GT_AREA[is_in_boxes==0]=INF
        locations_TO_GT_AREA[is_called_in_the_level==0]=INF

        locations_to_min_aera=K.min(locations_TO_GT_AREA,axis=1)#(n,)
        locations_to_gt_inds=K.argmin(locations_TO_GT_AREA,axis=1)#(n,) n中每一个值都在0到m-1之间

        all_point_in_image=np.ones((len(locations.eval()),4))
        for i in range(len(locations.eval())):
            all_point_in_image[i]=reg_targets_per_im[i,locations_to_gt_inds[i]].eval()
        reg_targets_per_im=tf.convert_to_tensor(all_point_in_image)#(n,4)


        #all_point_in_image=np.ones((len(locations.eval()),len(labels_per_im.eval()))) 
        all_point_in_image=np.ones((len(locations.eval()))) 
        for i in range(len(locations.eval())):
            all_point_in_image[i]=labels_per_im[locations_to_gt_inds[i]].eval()
        labels_per_im=tf.convert_to_tensor(all_point_in_image)#(n,)


        background_location=locations_to_min_aera.eval()==INF
        labels_per_im=labels_per_im.eval()
        for i in range(len(background_location)):
            if background_location[i]:
                labels_per_im[i]=0
        labels_per_im=tf.convert_to_tensor(labels_per_im)

        labels.append(labels_per_im)
        reg_targets.append(reg_targets_per_im)

    return labels,reg_targets



class IOUloss(layer):
    def __init__(self):
        super(IOUloss,self).__init__()

    def build(self,input_shape):
        super(IOUloss,self).build()
    
    def call(self,pred,target,weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        targ_aera = (target_left + target_right) * \
                    (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = K.minimum(pred_left, target_left) + \
                      K.minimum(pred_right, target_right)
        h_intersect = K.minimum(pred_bottom, target_bottom) + \
                      K.minimum(pred_top, target_top)
        aera_intersect=w_intersect*h_intersect
        aera_union=targ_aera+pred_aera-aera_intersect
        
        iou = (aera_intersect+1.0) / (aera_union+1.0)
        loss = -K.log(iou)
        if weight is not None and weight.sum()>0:
            return K.sum(loss*weight)/K.sum(weight)
        else:
            assert K.count_params(loss)!=0
            return K.mean(loss)
    
    def compute_output_shape(self,input_shape):
        return (input_shape[1][0],input_shape[0][1],1)



class Sigmoidfocalloss(layer):
    def __init__(self,gamma,alpha):
        super(Sigmoidfocalloss,self).__init__()
        self.gamma=gamma
        self.alpha=alpha
    
    def build(self,input_shape):
        super(Sigmoidfocalloss,self).build()

    def call(self,logits,targets):
        num_classes=logits.shape[1]
        gamma=self.gamma
        alpha=self.alpha
        datatype=K.dtype(targets)
        class_range=K.arange(1,num_classes+1,dtype=datatype).reshape(1,-1)#数值为1到num_classes 
        #维度为（1，num_classes)

        t=K.expand_dims(targets,1)#（b，1）
        p=K.sigmoid(logits)#(b,c) b张图片 c类

        term1=(1-p)**gamma*K.log(p)
        term2=p**gamma*K.log(1-p)
        loss= -K.cast((t==class_range),K.dtype(p))*term1*alpha\
            -K.cast((t!=class_range)*(t>=0),K.dtype(p))*term2*(1-alpha)
        return loss.sum()
    
    def compute_output_shape(self,input_shape):
        return (1)












      









