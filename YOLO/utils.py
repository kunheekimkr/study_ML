import tensorflow as tf
import numpy as np
import cv2
import colorsys
from operator import itemgetter

'''
Intersection Of Union (IoU)
Args:
    yolo_pred_boxes: 4D Tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] : (x_center, y_center, box_w, box_h)
    ground_truth_boxes: 1D Tensor[4] : (x_center,y_center,box_w,box_h)
Return:
    iou: 3D Tensor[CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
'''
def iou(yolo_pred_boxes, ground_truth_boxes):
    pred_boxes = yolo_pred_boxes
    pred_boxes = tf.stack([pred_boxes[:,:,:,0] - pred_boxes[:,:,:,2]/2, pred_boxes[:,:,:,1] - pred_boxes[:,:,:,3]/2, pred_boxes[:,:,:,0] + pred_boxes[:,:,:,2]/2, pred_boxes[:,:,:,1] + pred_boxes[:,:,:,3]/2])
    pred_boxes = tf.transpose(pred_boxes, [1,2,3,0])

    true_boxes = ground_truth_boxes
    true_boxes = tf.stack([true_boxes[0] - true_boxes[2]/2, true_boxes[1] - true_boxes[3]/2, true_boxes[0] + true_boxes[2]/2, true_boxes[1] + true_boxes[3]/2])
    true_boxes = tf.cast(true_boxes, tf.float32)

    left_up = tf.maximum(pred_boxes[:,:,:,0:2], true_boxes[0:2])
    right_down = tf.minimum(pred_boxes[:,:,:,2:], true_boxes[2:])
    intersection = right_down - left_up
    mask = tf.cast(intersection[:,:,:,0]>0, tf.float32) * tf.cast(intersection[:,:,:,1]>0, tf.float32) # 교집합이 존재하지 않는 경우 0
    
    inter_square = mask * intersection[:,:,:,0] * intersection[:,:,:,1]
    pred_box_square = (pred_boxes[:,:,:,2]- pred_boxes[:,:,:,0]) * (pred_boxes[:,:,:,3]- pred_boxes[:,:,:,1])
    true_box_square = (true_boxes[2] - true_boxes[0]) * (true_boxes[3] - true_boxes[1])
    return inter_square / (pred_box_square + true_box_square - inter_square + 1e-10)


'''
YOLO format data to Bounding Box Info
'''
def yolo_format_to_bounding_box_info(x_center, y_center, box_w, box_h, class_name, confidence):
    
    bounding_box_info={}
    bounding_box_info['left'] = int(x_center - (box_w/2))
    bounding_box_info['top'] = int(y_center - (box_h/2))
    bounding_box_info['right'] = int(x_center + (box_w/2))
    bounding_box_info['bottom'] = int(y_center + (box_h/2))
    bounding_box_info['class_name'] = class_name
    bounding_box_info['confidence'] = confidence

    return bounding_box_info

'''
Find Bounding Box with Highest Confidence
'''
def max_confidence_bounding_box(bounding_box_info_list):
    bounding_box_info_list_sorted = sorted(bounding_box_info_list, key=itemgetter('confidence'), reverse=True)
    return bounding_box_info_list_sorted[0]



'''
Draw Bounding Box and Label Info
'''
def draw(frame, x_min, y_min, x_max, y_max, label, confidence, color):
    #Draw Bounding Box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)

    #Draw Label Info
    text = label + ' ' + str('%.3f' % confidence)
    cv2.putText(frame, text, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


'''
Generate Colors
'''
def generate_color(num_classes):
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    np.random.seed(123)  
    np.random.shuffle(colors) 
    np.random.seed(None)  # Reset seed to default.
    return colors
