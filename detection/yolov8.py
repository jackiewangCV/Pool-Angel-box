import numpy as np
import cv2
import yaml

dd=yaml.safe_load(open('./data/coco128.yaml'))

CLASSES = dd['names']

class Yolov8:
    def __init__(self, modelPath, prob_threshold=0.35, iou_threshold=0.6, backend_id=0, target_id=0):
        self.model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(modelPath)
        self.prob_threshold=prob_threshold
        self.iou_threshold=iou_threshold
        self.backend_id=backend_id
        self.target_id=target_id
        self.model.setPreferableBackend(self.backend_id)
        self.model.setPreferableTarget(self.target_id)


    def detect(self, original_image):

        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Calculate scale factor
        scale = length / 640

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        self.model.setInput(blob)

        # Perform inference
        outputs = self.model.forward()

        outputs = np.array([cv2.transpose(outputs[0])])
    
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []


        #  x1,y1,x2,y2=int(r[0]),int(r[1]), int(r[2]),int(r[3])
        #     im_small=frame[y1:y2,x1:x2,:]
        
        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25 and maxClassIndex==0: ## filter only the persons
                # box = [
                #     outputs[0][i][0]*scale,
                #     outputs[0][i][1]*scale,
                #     outputs[0][i][0]*scale+outputs[0][i][2]*scale,
                #     outputs[0][i][1]*scale+outputs[0][i][3]*scale,
                #     ]
                box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), 
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], 
                outputs[0][i][3]
                ]
                box=[box[0]*scale, box[1]*scale, (box[0]+box[2])*scale, (box[1]+box[3])*scale]

                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                'confidence': scores[index],
                'box': box,
                }
            detections.append(detection)

        return detections

