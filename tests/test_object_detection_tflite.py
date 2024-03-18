import tflite_runtime.interpreter as tflite
import numpy as np
import cv2

from PIL import Image, ImageOps
import yaml 


dd=yaml.safe_load(open('./data/coco128.yaml'))

CLASSES = dd['names']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


class yolov5_tflite:

    def __init__(self,weights = 'yolov5s-fp16.tflite',image_size = 416,conf_thres=0.25,iou_thres=0.45):

        self.weights = weights
        self.image_size = image_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.interpreter = tflite.Interpreter(self.weights)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect(self,image):
        original_size = image.shape[:2]
        input_data = np.ndarray(shape=(1, self.image_size, self.image_size, 3), dtype=np.float32)
        #image = cv2.resize(image,(self.image_size,self.image_size))
        #input_data[0] = image.astype(np.float32)/255.0
        input_data[0] = image
        
        #self.interpreter.allocate_tensors()

        # Get input and output tensors
        #input_details = self.interpreter.get_input_details()
        #output_details = self.interpreter.get_output_details()


        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.output_details[0]['index'])
        pred=np.array([cv2.transpose(pred[0])])

        return pred

img_size=640

yolov5_tflite_obj = yolov5_tflite("./data/yolov8s_float16.tflite",img_size,0.25,0.5)



original_image: np.ndarray = cv2.imread("/data2/Freelancer/PoolAngel/Data/PoolAngel2/vid2-008.jpg")
# original_image = Image.open("/data2/Freelancer/PoolAngel/Data/PoolAngel2/vid2-008.jpg")
[height, width, _] = original_image.shape

# Prepare a square image for inference
length = max((height, width))
image = np.zeros((length, length, 3), np.uint8)
image[0:height, 0:width] = original_image

# Calculate scale factor
scale = length / 640

# Preprocess the image and prepare blob for model
blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)

normalized_image_array=np.array(blob)
print(normalized_image_array.shape)

outputs = yolov5_tflite_obj.detect(normalized_image_array.transpose(0,2,3,1))

rows = outputs.shape[1]

boxes = []
scores = []
class_ids = []

# Iterate through output to collect bounding boxes, confidence scores, and class IDs
for i in range(rows):
    classes_scores = outputs[0][i][4:]
    (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
    if maxScore >= 0.5:
        box = [
            outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
            outputs[0][i][2], outputs[0][i][3]]
        
        box=[box[0]*640,box[1]*640,box[2]*640,box[3]*640]

        boxes.append(box)
        scores.append(maxScore)
        class_ids.append(maxClassIndex)

# Apply NMS (Non-maximum suppression)
result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

detections = []
# scale=4.2

# Iterate through NMS results to draw bounding boxes and labels
for i in range(len(result_boxes)):
    index = result_boxes[i]
    box = boxes[index]
    detection = {
        'class_id': class_ids[index],
        'class_name': CLASSES[class_ids[index]],
        'confidence': scores[index],
        'box': box,
        'scale': scale}
    detections.append(detection)
    draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                        round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))

print(detections)

# Display the image with bounding boxes
cv2.imshow('image', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()