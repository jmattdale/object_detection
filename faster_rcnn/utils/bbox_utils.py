import numpy as np

def bboxes_relative(bboxes_absolute: np.array, image_shape: list):
    '''
    Args:
        bboxes_absolute: bboxes formatted as [xmin, ymin, xmax, ymax] integers
                         describing top-left and bottom-right pixel locations 
                         of image
        image_shape: shape of image in [height, width, channels] format
    Returns:
        bboxes_relative: bboxes formatted as [xmin, ymin, xmax, ymax] floats
                         describing top-left and bottom-right pixel relative
                         to the height and width of the image
    '''
    H, W, C = image_shape
    return np.stack([
        bboxes_absolute[..., 0] / W,
        bboxes_absolute[..., 1] / H,
        bboxes_absolute[..., 2] / W,
        bboxes_absolute[..., 3] / H
    ])


def convert_to_corners(boxes):
    return np.concatenate([
        boxes[..., :2] - boxes[..., 2:] / 2.0,
        boxes[..., :2] + boxes[..., 2:] / 2.0
    ], axis=-1)

def convert_to_xywh(boxes):
    '''
    Args:
        boxes in [x1,y1,x2,y2] format
    Returns:
        boxes in [cx, cy, w, h] format
    '''
    return np.concatenate([
        (boxes[..., :2] + boxes[..., 2:]) / 2.0,
        boxes[..., 2:] - boxes[..., :2]
    ], axis=-1)