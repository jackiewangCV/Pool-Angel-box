import tensorflow as tf

model_path='data/models/crowdhuman_yolov5m.pt'

# # Convert the TensorFlow model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model('data/models/crowdhuman_yolov5m.pb') # path to the SavedModel directory

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]


tflite_model = converter.convert()

# # Save the TensorFlow Lite model to a file
with open(model_path.replace('pt','tflite'), 'wb') as f:
    f.write(tflite_model)

