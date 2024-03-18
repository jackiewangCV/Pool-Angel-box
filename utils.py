import time
import cv2 as cv
import numpy as np

class BenchMark:
    def __init__(self) -> None:
        self.tasks={}
    
    def start(self, name):
        self.tasks[name]=time.time()

    def stop(self, name):
        tm=time.time()-self.tasks[name]
        print(f'{name} took {tm} seconds')
        return tm
    
classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

def letterbox(srcimg, target_size=(416, 416)):
    img = srcimg.copy()

    top, left, newh, neww = 0, 0, target_size[0], target_size[1]
    if img.shape[0] != img.shape[1]:
        hw_scale = img.shape[0] / img.shape[1]
        if hw_scale > 1:
            newh, neww = target_size[0], int(target_size[1] / hw_scale)
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            left = int((target_size[1] - neww) * 0.5)
            img = cv.copyMakeBorder(img, 0, 0, left, target_size[1] - neww - left, cv.BORDER_CONSTANT, value=0)  # add border
        else:
            newh, neww = int(target_size[0] * hw_scale), target_size[1]
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            top = int((target_size[0] - newh) * 0.5)
            img = cv.copyMakeBorder(img, top, target_size[0] - newh - top, 0, 0, cv.BORDER_CONSTANT, value=0)
    else:
        img = cv.resize(img, target_size, interpolation=cv.INTER_AREA)

    letterbox_scale = [top, left, newh, neww]
    return img, letterbox_scale

def unletterbox(bbox, original_image_shape, letterbox_scale):
    ret = bbox.copy()

    h, w = original_image_shape
    top, left, newh, neww = letterbox_scale

    if h == w:
        ratio = h / newh
        ret = ret * ratio
        return ret

    ratioh, ratiow = h / newh, w / neww
    ret[0] = max((ret[0] - left) * ratiow, 0)
    ret[1] = max((ret[1] - top) * ratioh, 0)
    ret[2] = min((ret[2] - left) * ratiow, w)
    ret[3] = min((ret[3] - top) * ratioh, h)

    return ret.astype(np.int32)

def vis(preds, res_img, letterbox_scale=None, fps=None):
    ret = res_img.copy()

    # draw FPS
    if fps is not None:
        fps_label = "FPS: %.2f" % fps
        cv.putText(ret, fps_label, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # draw bboxes and labels
    for pred in preds:
        bbox = pred.bbox[-1]
        conf = pred.confidence
        dist=pred.dist_pool
        ratio= pred.ratio
        kpts= pred.pose

        # bbox
        # if letterbox_scale is not None:
        #     xmin, ymin, xmax, ymax = unletterbox(bbox, ret.shape[:2], letterbox_scale)
        # else:
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if pred.dist_pool==0:
            clr=(0,0,255)
        elif pred.warn:
            clr=(30,255,255)
        else:
            clr=(0, 255, 0)

        cv.rectangle(ret, (xmin, ymin), (xmax, ymax), clr, thickness=2)
        cv.circle(ret, (round(0.5*(xmax+xmin)), ymax), 5, (0, 0, 255), thickness=-1)

        # for i in range(kpts.shape[0]):
        #     cv.circle(ret, (round(kpts[i,0]+xmin), round(round(kpts[i,1]+ymin))), 2, (255, 0, 0), thickness=-1)

        # label
        label = "{:.2f}".format(ratio*100)
        cv.putText(ret, label, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

    return ret


