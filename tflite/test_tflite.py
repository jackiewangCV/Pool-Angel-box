
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

## Test object detection model  ###

model_file='data/models/crowdhuman_yolov5m.tflite'

interpreter = tflite.Interpreter(model_path=model_file)

interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
image=cv2.imread('data/images/vid2-005.jpg')
image=cv2.resize(image, (640,640))

input_data = image[:,:,::-1].transpose([2,0,1])
input_data=input_data.astype('float32')[None]
print(input_data.shape)


interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[3]['index'])
pred=output_data[0]

from pose.utils.boxes import letterbox, scale_boxes, non_max_suppression, xyxy2xywh

pred = non_max_suppression(torch.tensor(pred), 0.45, 0.2, classes=0)
print(pred)
#print(pred.shape)

for i in range(len(pred)):
    vals=pred[i]
    bbox=vals[:4]
    bbox=bbox.astype('int')
    print(image.shape, bbox)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
    if i>100:
        break

plt.imshow(image)

plt.show()