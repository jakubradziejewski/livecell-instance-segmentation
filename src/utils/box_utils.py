import torch


def encode_boxes(boxes, anchors):
    """Encode bounding boxes relative to anchor boxes."""
    anchors_ctr_x = (anchors[:, 0] + anchors[:, 2]) / 2.0
    anchors_ctr_y = (anchors[:, 1] + anchors[:, 3]) / 2.0
    anchors_w = anchors[:, 2] - anchors[:, 0]
    anchors_h = anchors[:, 3] - anchors[:, 1]

    boxes_ctr_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    boxes_ctr_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    boxes_w = boxes[:, 2] - boxes[:, 0]
    boxes_h = boxes[:, 3] - boxes[:, 1]

    anchors_w = torch.clamp(anchors_w, min=1.0)
    anchors_h = torch.clamp(anchors_h, min=1.0)
    boxes_w = torch.clamp(boxes_w, min=1.0)
    boxes_h = torch.clamp(boxes_h, min=1.0)

    dx = (boxes_ctr_x - anchors_ctr_x) / anchors_w
    dy = (boxes_ctr_y - anchors_ctr_y) / anchors_h
    dw = torch.log(boxes_w / anchors_w)
    dh = torch.log(boxes_h / anchors_h)

    deltas = torch.stack([dx, dy, dw, dh], dim=1)

    return deltas



def clip_boxes_to_image(boxes, image_size):
    """Clip boxes to image boundaries."""
    h, w = image_size
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
    return boxes

def filter_small_boxes(boxes, min_size=1):
    """Remove boxes smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    return keep