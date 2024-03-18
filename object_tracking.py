import cv2 as cv
from tracking import dasiamrpn


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



class ObjectTracking:
    def __init__(self, tracker='dasiamrpn') -> None:
        if tracker=='dasiamrpn':
            self.tracker=TrackerDasiamrpn()
        else:
            NotImplementedError(f"{tracker} not implemented ")
    
    def init(self, last_frame, roi):
        self.tracker.init(last_frame, roi)

    def track(self, image):
        preds=self.tracker.track(image)
        return preds
    

class TrackerDasiamrpn:
    def __init__(self,model_path=None) -> None:
        backend_id = backend_target_pairs[BACKEND_TARGET][0]
        target_id = backend_target_pairs[BACKEND_TARGET][1]

        # Instantiate DaSiamRPN
        self.model = dasiamrpn.DaSiamRPN(
            kernel_cls1_path='./data/object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx',
            kernel_r1_path='./data/object_tracking_dasiamrpn_kernel_r1_2021nov.onnx',
            model_path='./data/object_tracking_dasiamrpn_model_2021nov.onnx',
            backend_id=backend_id,
            target_id=target_id)
        
    def init(self, last_frame, roi):
        self.model.init(last_frame, roi)

    def track(self, frame):
        isLocated, bbox, score = self.model.infer(frame)
        return isLocated, bbox, score
