import cv2 as cv
from detection import nanodet, mp_persondet, yolov8, yolox
import utils
from property import Person


CONFIDENCE=0.25
NMS_THR=0.6
BACKEND_TARGET=0
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]



class ObjectDetection:
    def __init__(self, detector='nanodet', tracker=None) -> None:
        if detector=='nanodet':
            self.detector=ObjectDetectionNanoDet('./data/object_detection_nanodet_2022nov.onnx')
        elif detector=='nanodet_int8':
            BACKEND_TARGET=3
            print('Using NPU backend for object detection')
            self.detector=ObjectDetectionNanoDet('./data/object_detection_nanodet_2022nov_int8.onnx')
        elif detector=='yolov8':
            self.detector=ObjectDetectionYolov8('./data/yolov8s.onnx')
            print('Using yolo-v8 for object detection')
        elif detector=='mpdet':
            self.detector=ObjectDetectionMpDet('./data/person_detection_mediapipe_2023mar.onnx')
        elif detector=='yolox':
            print('Using yolox for object detection')
            self.detector=ObjectDetectionYoloX("./data/object_detection_yolox_2022nov.onnx")
        elif detector=='yolox_int8':
            print('Using yolox for object detection int8 npu backend')
            self.detector=ObjectDetectionYoloX("./data/object_detection_yolox_2022nov_int8.onnx")
        else:
            NotImplementedError(f"{detector} not implemented ")
        
        self.tracker=None

    def detect(self, image):
        preds=self.detector.detect(image)
        return preds
    

class ObjectDetectionYolov8:
    def __init__(self, model_path) -> None:
        backend_id = backend_target_pairs[BACKEND_TARGET][0]
        target_id = backend_target_pairs[BACKEND_TARGET][1]

        self.model = yolov8.Yolov8(modelPath= model_path,
                    prob_threshold=CONFIDENCE,
                    iou_threshold=NMS_THR,
                    backend_id=backend_id,
                    target_id=target_id)
    
    def detect(self, image):
        detections=self.model.detect(image)
        persons=[]

        for inds,det in enumerate(detections):
            persons.append(Person(inds,bbox=det['box'],confidence=det['confidence']))
            # print(det['box'])
        return persons
        

class ObjectDetectionNanoDet:
    def __init__(self, model_path, vis=True) -> None:
        backend_id = backend_target_pairs[BACKEND_TARGET][0]
        target_id = backend_target_pairs[BACKEND_TARGET][1]

        self.model = nanodet.NanoDet(modelPath= model_path,
                    prob_threshold=CONFIDENCE,
                    iou_threshold=NMS_THR,
                    backend_id=backend_id,
                    target_id=target_id)
        self.vis=vis

    def detect(self, image):
        input_blob = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Letterbox transformation
        input_blob, letterbox_scale = utils.letterbox(input_blob)
        # Inference
        preds = self.model.infer(input_blob)
        preds=preds[preds[:,-1]==0,:]  ## Filter only persons.
        persons=[]

        for inds,p in enumerate(preds):
            persons.append(Person(inds,p[:4],confidence=p[-1]))

        return persons



class ObjectDetectionYoloX:
    def __init__(self, model_path) -> None:
        backend_id = backend_target_pairs[BACKEND_TARGET][0]
        target_id = backend_target_pairs[BACKEND_TARGET][1]

        self.model = yolox.YoloX(modelPath= model_path,
                    confThreshold=CONFIDENCE,
                    nmsThreshold=NMS_THR,
                    targetId=target_id)

    def detect(self, image):
        input_blob = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Letterbox transformation
        input_blob, letterbox_scale = utils.letterbox(input_blob, target_size=(640,640))
        # print(letterbox_scale)
        # Inference
        preds = self.model.infer(input_blob)
        if len(preds)>0:
            preds=preds[preds[:,-1]==0,:]  ## Filter only persons.
        persons=[]

        for inds,p in enumerate(preds):
            xmin, ymin, xmax, ymax = utils.unletterbox(p[:4], image.shape[:2], letterbox_scale)
            bbox=[xmin, ymin, xmax, ymax]
            persons.append(Person(inds,bbox,confidence=p[-1]))

        return persons



class ObjectDetectionMpDet:
    def __init__(self, model_path, vis=True) -> None:
        backend_id = backend_target_pairs[BACKEND_TARGET][0]
        target_id = backend_target_pairs[BACKEND_TARGET][1]
        self.model = mp_persondet.MPPersonDet(modelPath=model_path,
                            nmsThreshold=CONFIDENCE,
                            scoreThreshold=0.5,
                            topK=1,
                            backendId=backend_id,
                            targetId=target_id)
        self.vis=vis

    def detect(self, image):
        persons = self.model.infer(image)
        return image, persons
    
