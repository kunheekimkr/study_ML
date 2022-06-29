import tensorflow as tf
import numpy as np

def bounds_per_dimension(ndarray):
    return map(
        lambda e: range(e.min(), e.max() + 1),
        np.where(ndarray!=0)
    )

def zero_trim_ndarray(ndarray):
    return ndarray[np.ix_(*bounds_per_dimension(ndarray))]


'''
Process Ground-Truth data for YOLO Format
Reference :  https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/object_detection/voc.py#L115
  Args:
    original_image : (original_height, orignal_width, channel) image tensor
    bbox : (max_object_num_in_batch, 4) = (ymin / height, xmin / width, ymax / height, xmax / width)
    class_labels : (max_object_num_in_batch) = class labels without one-hot-encoding
    input_width 
    input_height 
  Returns:
    image: (resized_height, resized_width, channel) image ndarray
    labels: 2-D list [object_num, 5] (xcenter (Absolute Coordinate), ycenter (Absolute Coordinate), w (Absolute Coordinate), h (Absolute Coordinate), class_num)
    object_num: total object number in image
'''
def process_each_ground_truth(original_image, bbox, class_labels,input_width,input_height):
    image = original_image.numpy()
    image = zero_trim_ndarray(image)

    original_height, original_width = image.shape[:2]
    width_rate = input_width *1.0 / original_width
    height_rate = input_height *1.0 / original_height

    image= tf.image.resize(image, [input_height, input_width])

    object_num = np.count_nonzero(bbox, axis=0)[0]
    labels = [[0,0,0,0,0]] * object_num
    for i in range(object_num):
        x_min = bbox[i][1] * original_width
        y_min = bbox[i][0] * original_height
        x_max = bbox[i][3] * original_width
        y_max = bbox[i][2] * original_height

        class_num = class_labels[i]

        x_center = (x_min + x_max)*1.0 / 2.0 * width_rate
        y_center = (y_min + y_max)*1.0 / 2.0 * height_rate

        box_width = (x_max - x_min) * width_rate
        box_height = (y_max - y_min) * height_rate
        labels[i] = [x_center, y_center, box_width, box_height, class_num]

    return [image.numpy(), labels, object_num]