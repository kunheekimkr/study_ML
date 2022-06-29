import tensorflow as tf
import numpy as np
from utils import iou

'''
Loss function of YOLO
Args:
    predict: 3D Tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL*5 + NUM_CLASSES]
    lables: 2D Tensor [object_num, 5] (x_center, y_center, box_w, box_h, class_num) (Absolute Coordinate)
    each_object_num: each object number in image
    num_classes
    boxes_per_cell
    cell_size
    input_width
    input_height
    coord_scale : coefficient for coordinate loss
    object_scale : coefficient for object loss
    noobject_scale : coefficient for noobject loss
    class_scale : coefficient for class loss

Retuns:
    total_loss: coordinate loss + object_loss + noobject_loss + class_loss
    coord_loss
    object_loss
    noobject_loss
    class_loss
'''

def yolo_loss(predict, labels, each_object_num, num_classes,
              boxes_per_cell, cell_size, input_width, input_height,
              coord_scale, object_scale, noobject_scale, class_scale):
    
    # parse coordinate vector
    predict_boxes = tf.reshape(predict[:, :, num_classes + boxes_per_cell :], [cell_size, cell_size, boxes_per_cell, 4])

    pred_x_center = predict_boxes[:,:,:,0]
    pred_y_center = predict_boxes[:,:,:,1]
    pred_sqrt_w = tf.cast(tf.sqrt(tf.minimum(input_width*1.0, tf.maximum(0.0, predict_boxes[:,:,:,2]))), tf.float32)
    pred_sqrt_h = tf.cast(tf.sqrt(tf.minimum(input_height*1.0, tf.maximum(0.0, predict_boxes[:,:,:,3]))),tf.float32)

    # parse label
    labels =np.array(labels).astype('float32')
    label = labels[each_object_num, :]
    label_x_center = label[0]
    label_y_center = label[1]
    label_sqrt_w = tf.sqrt(label[2])
    label_sqrt_h = tf.sqrt(label[3])

    # calculate IoU
    iou_predict_truth = iou(predict_boxes, label[0:4])

    I = iou_predict_truth
    max_I = tf.reduce_max(I, 2, keepdims=True)
    best_box_mask = tf.cast((I>=max_I), tf.float32)

    C = iou_predict_truth
    pred_C = predict[:,:,num_classes:num_classes + boxes_per_cell]

    P =tf.one_hot(tf.cast(label[4], tf.int32), num_classes, dtype=tf.float32)
    pred_P = predict[:,:,0:num_classes]

    object_exists_cell = np.zeros([cell_size, cell_size, 1])
    object_exists_cell_i, object_exists_cell_j = int(cell_size * label_y_center / input_height), int(cell_size * label_x_center / input_width)
    object_exists_cell[object_exists_cell_i][object_exists_cell_j] = 1

    # calculate loss
    coord_loss = (tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_x_center - label_x_center) / input_width /cell_size) +
                  tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_y_center - label_y_center) / input_height /cell_size) +
                    tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_w - label_sqrt_w)) / input_width +
                    tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_h - label_sqrt_h)) / input_height) * coord_scale

    object_loss = tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_C - C)) * object_scale
    
    noobject_loss = tf.nn.l2_loss((1 - object_exists_cell) * (pred_C)) * noobject_scale
    
    class_loss = tf.nn.l2_loss(object_exists_cell * (pred_P - P)) * class_scale
    
    total_loss = coord_loss + object_loss + noobject_loss + class_loss

    return total_loss, coord_loss, object_loss, noobject_loss, class_loss