from onnx import numpy_helper
import onnx

MODEL_PATH = "data/yolov8n.onnx"
_model = onnx.load(MODEL_PATH)
INTIALIZERS=_model.graph.initializer
Weight=[]
for initializer in INTIALIZERS:
    W= numpy_helper.to_array(initializer)
    Weight.append(W)
    print(W)


