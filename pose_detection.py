import cv2 as cv
from detection import mp_pose
import utils
from property import Person
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
import matplotlib.patches as patches

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





class PoseDetection:
    def __init__(self, detector='movenet') -> None:
        if detector=='mediapipe':
            self.pose_detector=PoseDetection_MP('/data2/Freelancer/PoolAngel/main/data/pose_estimation_mediapipe_2023mar.onnx')
        elif detector=='movenet':
            self.pose_detector=PoseDetection_movenet("./data/lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite")
        else:
            NotImplementedError(f"{detector} not implemented ")
        

    def detect(self, image, persons):
        preds=self.pose_detector.detect(image, persons)
        return preds




class PoseDetection_MP:
    def __init__(self, model_path, vis=True) -> None:
        backend_id = backend_target_pairs[BACKEND_TARGET][0]
        target_id = backend_target_pairs[BACKEND_TARGET][1]
        self.model = mp_pose.MPPose(modelPath= model_path,
                    prob_threshold=CONFIDENCE,
                    iou_threshold=NMS_THR,
                    backend_id=backend_id,
                    target_id=target_id)
        self.vis=vis

    def detect(self, image, persons=None):
        input_blob = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Letterbox transformation
        input_blob, letterbox_scale = utils.letterbox(input_blob)
        # Inference
        preds = self.model.infer(input_blob)
        preds=preds[preds[:,-1]==0,:]  ## Filter only persons.
        persons=[]

        for inds,p in enumerate(preds):
            persons.append(Person(inds,p[:4],confidence=p[-1]))

        img=[]

        if self.vis:
            img = utils.vis(preds, image, letterbox_scale)

        return persons
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

def compute_distances(pnt1,pnt2):
    return np.sqrt((pnt1[0]-pnt2[0])**2+(pnt1[1]-pnt2[1])**2)

class PoseDetection_movenet:
    def __init__(self, model_path) -> None:
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = 256
    
    def compute_rat(self, kpts):
        head_len=compute_distances(kpts[KEYPOINT_DICT['left_ear']],kpts[KEYPOINT_DICT['right_ear']])
        person_height1=compute_distances(kpts[KEYPOINT_DICT['left_wrist']],kpts[KEYPOINT_DICT['left_elbow']])+compute_distances(kpts[KEYPOINT_DICT['left_elbow']],kpts[KEYPOINT_DICT['left_shoulder']])+compute_distances(kpts[KEYPOINT_DICT['left_shoulder']],kpts[KEYPOINT_DICT['right_shoulder']])\
                        +compute_distances(kpts[KEYPOINT_DICT['right_shoulder']],kpts[KEYPOINT_DICT['right_elbow']])+compute_distances(kpts[KEYPOINT_DICT['right_elbow']],kpts[KEYPOINT_DICT['right_wrist']])
        

        person_height2=compute_distances(kpts[KEYPOINT_DICT['left_ear']],kpts[KEYPOINT_DICT['left_shoulder']])+compute_distances(kpts[KEYPOINT_DICT['left_shoulder']],kpts[KEYPOINT_DICT['left_hip']])\
            +compute_distances(kpts[KEYPOINT_DICT['left_hip']],kpts[KEYPOINT_DICT['left_knee']])+compute_distances(kpts[KEYPOINT_DICT['left_knee']],kpts[KEYPOINT_DICT['left_ankle']])
        

        person_height3=compute_distances(kpts[KEYPOINT_DICT['right_ear']],kpts[KEYPOINT_DICT['right_shoulder']])+compute_distances(kpts[KEYPOINT_DICT['right_shoulder']],kpts[KEYPOINT_DICT['right_hip']])\
            +compute_distances(kpts[KEYPOINT_DICT['right_hip']],kpts[KEYPOINT_DICT['right_knee']])+compute_distances(kpts[KEYPOINT_DICT['right_knee']],kpts[KEYPOINT_DICT['right_ankle']])
        
        person_height=max([person_height1,person_height2,person_height3]) #
          
        rat=head_len/person_height

        return rat


    def detect_one(self, image):
        input_image = tf.expand_dims(image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)

        input_image = tf.cast(input_image, dtype=tf.uint8)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_image.numpy())
        # Invoke inference.
        self.interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        return keypoints_with_scores

    def detect(self, frame, persons):
        height, width, _ = frame.shape
        for i,p in enumerate(persons):
            r=p.bbox[-1]
            x1,y1,x2,y2=int(r[0]),int(r[1]), int(r[2]),int(r[3])
            im_small=frame[y1:y2,x1:x2,:]

            kpts=self.detect_one(im_small)
            kpts=np.squeeze(kpts)

            kpts[:, 1] = kpts[:, 1]*width
            kpts[:, 0] = kpts[:, 0]*height

            # 
            # print(kpts)
            
            # print(kpts*width)
            # print(kpts*height)

            # kpts[:,0]=kpts[:,1]*height
            # kpts[:,1]=kpts[:,2]*width
            # kpts=kpts.astype('int32')
            
            # for l in range(kpts.shape[0]):
            #     cv.circle(im_small, (round(kpts[l,0]), round(kpts[l,1])), 5, (255, 0, 0), thickness=-1)
            # plt.imshow(im_small[:,:,::-1])
            # plt.show()

            # output_overlay = draw_prediction_on_image(im_small, kpts)
            # plt.figure(figsize=(5, 5))
            # plt.imshow(output_overlay)
            # _ = plt.axis('off')
            # plt.show()

            rat=self.compute_rat(kpts)
            persons[i].update_infos(poses=kpts,ratio=rat)
        
        return persons


##########################################################################################################################################################
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(output_image_width, output_image_height),
         interpolation=cv2.INTER_CUBIC)
  return image_from_plot

