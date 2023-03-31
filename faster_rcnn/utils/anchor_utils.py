import numpy as np

def calc_wh(base_size, scale, ar):
    base_size *= scale
    ar = ar[0]/ar[1]
    height = base_size / ar
    width = ar * base_size
    return width, height 

def create_anchors(image_size, stride=100, aspect_ratios=[[1,1], [1,2], [2,1]], scales=[0.25, 0.5, 1], base_anchor_size=128):
    x = np.arange(start=stride/2, stop=image_size[0], step=stride, dtype=float)
    y = np.arange(start=stride/2, stop=image_size[1], step=stride, dtype=float)
    xv, yv = np.meshgrid((x), (y))
    centers = np.stack([yv, xv], axis=-1)

    anchors = None
    for scale in scales:
        for ar in aspect_ratios:
            width, height = calc_wh(base_anchor_size, scale, ar)
            w = np.ones([centers.shape[0], centers.shape[1], 1]) * width
            h = np.ones([centers.shape[0], centers.shape[1], 1]) * height
            wh = np.concatenate([w,h], axis=-1)
            
            anchors_grid = np.concatenate([centers, wh], axis=-1)
            anchors_flat = anchors_grid.reshape(anchors_grid.shape[0] * anchors_grid.shape[1], anchors_grid.shape[2])

            if anchors is None:
                anchors = anchors_flat
            else:
                anchors = np.concatenate([anchors, anchors_flat], axis=0)
    return anchors