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




def fcospostprocessor(cfg,locations,box_cls,box_regression,centerness,image_sizes):

    
    pre_nms_thres =cfg.MODEL.FCOS.INFERENCE_TH
    pre_nms_top_n=cfg.MODEL.FCOS.PRE_NMS_TOP_N
    nms_thresh=cfg.MODEL.FCOS.NMS_TH
    fpn_post_nms_top_n=cfg.TEST.DETECTIONS_PER_IMG
    num_classes=cfg.MODEL.FCOS.NUM_CLASSES
    to_remove=1
    min_size=0
    
    sampled_boxes=[]
    for _,(l,o,b,c) in enumerate(zip(locations,box_cls,box_regression,centerness)):
        sampled_boxes.append(
            postprocess_for_single_feature_map(
                l,o,b,c,image_sizes,to_remove,min_size
            )
        )
    boxlists=list(zip(*sampled_boxes))
    all_boxes=[cat_boxlist(boxlist) for boxlist in boxlists]#boxlist一张图片的所有stage的bbox
    all_boxes=select_over_all_levels(all_boxes)

    return all_boxes




def postprocess_for_single_feature_map(
    locations,
    box_cls,
    box_regression,
    centerness,
    image_sizes,
    to_remove,
    min_size
    )：
    N,C=box_cls.shape[0],box_cls.shape[3]
    box_cls=K.sigmoid(box_cls.reshape(N,-1,C))
    box_regression = box_regression.reshape(N, -1, 4)
    centerness =K.sigmoid( centerness.reshape(N, -1))
    candicate_inds=box_cls>pre_nms_thres
    pre_nms=candicate_inds.reshape(N,-1)
    pre_nms_number=K.sum(pre_nms,axis=1)
    mask=K.cast([pre_nms_number<pre_nms_top_n],'int')
    pre_nms_number=mask*pre_nms_number+(1-mask)*np.ones((N,1))*pre_nms_top_n
    cls_score=box_cls*centerness[:,:,None]
    results=[]
    for i in range(N):
        per_candicate_inds=candicate_inds[i]#(wh,c) True or Fasle
        per_candicate_nozeros=np.nonzero(per_candicate_inds.eval())
        #返回所有非零值的参数,每个参数包含 位置范围（0-wh），类范围（0-80） 
        #例如(array([0, 0, 1]), array([0, 1, 1]))
        per_cls_score=cls_score[i]#(wh,c)
        per_cls_score=per_cls_score[per_candicate_inds]#(m,)
        per_class=per_candicate_nozeros[1]+1#array([1, 2, 2])
        per_box_loc=per_candicate_nozeros[0]#array([0, 0, 1])
        per_box_regression=box_regression[i]#(wh,4)
        per_box_regression=per_box_regression[per_box_loc]#(m,4)
        per_locations=locations[per_box_loc]#locations(wh,2),per_locations(m,2)
        per_pre_nms_number=pre_nms_number[i]#整数tensor

        if per_candicate_inds.sum().eval()>per_pre_nms_number.eval():#如果概率大于零的预测点数量大于1000
            per_cls_score,indice_topk=tf.nn.top_k(per_cls_score,per_pre_nms_number)
            #per_cls_score   (per_pre_nms_top_n,) 
            #indice_topk     (per_pre_nms_top_n，)
            per_class=per_class[indice_topk]#(per_pre_nms_top_n,) 
            per_locations=per_locations[indice_topk]#(per_pre_nms_top_n，4)
            per_box_regression=per_box_regression[indice_topk]#(per_pre_nms_top_n，2)

        detections=tf.stack([
            per_locations[:, 0] - per_box_regression[:, 0],
            per_locations[:, 1] - per_box_regression[:, 1],
            per_locations[:, 0] + per_box_regression[:, 2],
            per_locations[:, 1] + per_box_regression[:, 3],   
        ],axis=1)#(per_pre_nms_top_n，4)

        h,w=image_sizes[i].shape[0],image_sizes[i].shape[1]
        boxlist={}
        boxlist.update('image_size':(int(h),int(w)),'detections':detections,'labels':per_class,\
            'scores':per_cls_score)
        boxlist=clip_to_image(boxlist,to_remove,remove_empty=True)
        #boxlist = remove_small_boxes(boxlist,to_remove,min_size)
        results.append(boxlist)
    return results#所有图片在一个stage的特征图上的boxlist的集合


def cat_boxlist(boxlist):

    boxes={}
    h=boxlist[0]['image_size'][0]
    w=boxlist[0]['image_size'][1]
    assert all(bl_per_stage['image_size'][0]==h for bl_per_stage in boxlist)
    assert all(bl_per_stage['image_size'][1]==w for bl_per_stage in boxlist)
    def _cat(blist):
        if len(blist)==1:
            return list[0]
        return tf.concat(blist,dim=0)
    detections=_cat([bl_per_stage['detections'] for bl_per_stage in boxlist])
    labels=_cat([bl_per_stage['labels']] for bl_per_stage in boxlist)
    scores=_cat([bl_per_stage['scores'] for bl_per_stage in boxlist])
    boxes['detections']=detections
    boxes['labels']=labels
    boxes['scores']=scores
    boxes['image_sizes']=(int(h),int(w))
    return boxes


def select_over_all_levels(boxlist):
    num_images=len(boxlist)
    results=[]
    for i in range(num_images):
        boxes_per_image=boxlist[i]
        scores=boxes_per_image['scores'].reshape(-1,)
        labels=boxes_per_image['labels'].reshape(-1,)
        boxes=boxes_per_image['detections']
        image_size=boxes_per_image['image_size']
        result=[]
        for j in range(1,num_classes):
            inds=K.cast([labels==j],'int')
            inds=np.flatnonzero(inds.eval())
            scores_j=scores[inds]
            boxes_j=boxes[inds,:]
            boxes_for_class={}
            boxes_for_class['detections']=boxes_j
            boxes_for_class['scores']=scores_j
            boxes_for_class['image_size']=image_size
            boxes_for_class=boxlist_nms(boxes_for_class,nms_thresh)
            num_labels=boxes_for_class['detections'].shape[0]
            labels=np.ones((num_labels,))*j
            labels=K.variable(labels,dtype='int64')
            boxlist['labels']=labels
            result.append(boxes_for_class)
        result=cat_boxlist(result)

        number_of_detections=result['detections'].shape[0]
        if number_of_detections>fpn_post_nms_top_n>0:
            cls_score=result['scores']
            image_thresh=kth_value(cls_score.eval(),number_of_detections)
            keep=K.cast([cls_score>=image_thresh],'int')
            keep=np.flatnonzero(keep.eval())
            result['detections']=result['detections'][keep,:]
            result['scores']=result['scores'][keep]
            result['labels']=result['labels'][keep]
        results.append(result)

        return results#包含了所有图片信息的列表，每个图片信息在列表中以字典的形式储存


def kth_value(array,k):
    for i in range(1,k):
        for j in range(i,0,-1):
            if array[j]>array[j-1]:
                array[j],array[j-1]=array[j-1],array[j]
            else:
                pass
    for i in range(k,len(array)):
        if array[i]>array[k-1]:
            array[k-1]=array[i]
            for j in range(k-1,0,-1)：
                if array[j]>array[j-1]:
                    array[j],array[j-1]=array[j-1],array[j]
                else:
                    pass
    return[array[k-1]]



def clip_to_image(boxlist,to_remove,remove_empty=True):
    
    bbox=boxlist['detections']
    image_size=boxlist['image_size']
    h,w=image_size[0],image_size[1]
    for i in [0,2]:
        maskbot=K.cast([bbox[:,i]>0],'int')
        bbox[:,i]=maskbot*bbox[:,i]
        masktop=K.cast([bbox[:,i]<w-to_remove],'int')
        bbox[:,i]=masktop*bbox[:,i]+(1-masktop)*np.ones((bbox[:,i].shape))*(w-to_remove)
    for i in [1,3]:
        maskbot=K.cast([bbox[:,i]>0],'int')
        bbox[:,i]=maskbot*bbox[:,i]
        masktop=K.cast([bbox[:,i]<h-to_remove],'int')
        bbox[:,i]=masktop*bbox[:,i]+(1-masktop)*np.ones((bbox[:,i].shape))*(h-to_remove)
    if remove_empty:
        keep=(bbox[:,3]>bbox[:,1])&(bbox[:, 2] > bbox[:, 0])
        keep=keep.reshape(-1,1)
        keep=keep.repeat(4,axis=1)
        bbox=bbox*keep
        bbox_eval=bbox.eval()
        row_to_del=np.where(bbox_eval[:,0]==0)
        bbox_eval=np.delete(bbox_eval,list(row_to_del),0)
        bbox=K.variable(bbox_eval,dtype='float64')
    boxlist['detections']=bbox

    return boxlist



'''def remove_small_boxes(boxlist,to_remove,min_size):
    boxlist=convert_to_xywh(boxlist,to_remove)
    bbox=boxlist['detections']
    bbox_W=bbox[:,2]
    bbox_H=bbox[:,3]
    keep=(bbox_H>=min_size&bbox_W>=min_size)'''


def convert_to_xywh(boxlist,to_remove):
    
    bbox=boxlist['detections']
    bbox_W=bbox[:,2]-bbox[:,0]+to_remove
    bbox_H=bbox[:,3]-bbox[:,1]+to_remove
    bbox[:,2]=bbox_W
    bbox[:,3]=bbox_H
    boxlist['detections']=bbox
    return boxlist


def boxlist_nms(boxlist,nms_thresh,max_proposals=20):
    if nms_thresh<=0:
        return boxlist
    boxlist_new={}
    boxes=boxlist['detections']
    score=boxlist['scores']
    keep=tf.image.non_max_suppression(boxes=boxes,scores=score,max_output_size=max_proposals,\
                                  iou_threshold=nms_thresh)
    #boxes=tf.gather(boxes,keep)
    boxes=boxes[keep,:]
    score=score[keep,:]
    boxlist_new['detections']=boxes
    boxlist_new['scores']=score
    return boxlist

