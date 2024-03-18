from yolov5.models.experimental import attempt_load
import torch
import onnx
# import onnx_tf


device=torch.device('cpu')
model_path='data/models/crowdhuman_yolov5m.pt'

pytorch_model = attempt_load(model_path, map_location=device)

im_size=640

dummy_input = torch.rand((1, 3, im_size, im_size))

onnx_model_path = model_path.replace('pt','onnx')
torch.onnx.export(pytorch_model, dummy_input, onnx_model_path, export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=12,    # the ONNX version to export the model to 
         do_constant_folding=False,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
        ) 
print('onnx_expoted')

# # # Load the ONNX model
# onnx_model = onnx.load(onnx_model_path)

# # Convert the ONNX model to TensorFlow format
# tf_model_path = model_path.replace('pt','pb')
# tf_rep = onnx_tf.backend.prepare(onnx_model)
# tf_rep.export_graph(tf_model_path)
# print('tf model expoted')