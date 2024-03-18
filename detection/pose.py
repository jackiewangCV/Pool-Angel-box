import torch
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms as T

from pose.models import get_pose_model
from pose.utils.boxes import letterbox, scale_boxes, non_max_suppression, xyxy2xywh
from pose.utils.decode import get_final_preds, get_simdr_final_preds
from pose.utils.utils import setup_cudnn, get_affine_transform, draw_keypoints
from pose.utils.utils import VideoReader, VideoWriter, WebcamStream, FPS

import sys
sys.path.insert(0, 'yolov5')
from yolov5.models.experimental import attempt_load
import tflite_runtime.interpreter as tflite

use_tflite=True

class Pose:
    def __init__(self, 
        det_model,
        pose_model,
        img_size=640,
        conf_thres=0.25,
        iou_thres=0.45, 
    ) -> None:
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if use_tflite:
            model_file=f'../data/models/crowdhuman_yolov5m_{img_size}.tflite'
            self.interpreter = tflite.Interpreter(model_path=model_file)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.det_model = attempt_load(det_model, map_location=self.device)
            self.det_model = self.det_model.to(self.device)

        self.model_name = pose_model
        if use_tflite:
            model_file2=f'../data/models/posehrnet_w32_256x192.tflite'
            self.interpreter2 = tflite.Interpreter(model_path=model_file2)
            self.interpreter2.allocate_tensors()
            self.input_details2 = self.interpreter2.get_input_details()
            self.output_details2 = self.interpreter2.get_output_details()
        else:
            self.pose_model = get_pose_model(pose_model)

            self.pose_model.load_state_dict(torch.load(pose_model, map_location='cpu'))
            self.pose_model = self.pose_model.to(self.device)
            self.pose_model.eval()

        self.patch_size = (192, 256)

        self.pose_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.coco_skeletons = [
            [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
            [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
        ]


        


    def preprocess(self, image):
        img = letterbox(image, new_shape=self.img_size)
        print(img.shape)
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img[None]
        return img

    def box_to_center_scale(self, boxes, pixel_std=200):
        boxes = xyxy2xywh(boxes)
        r = self.patch_size[0] / self.patch_size[1]
        mask = boxes[:, 2] > boxes[:, 3] * r
        boxes[mask, 3] = boxes[mask, 2] / r
        boxes[~mask, 2] = boxes[~mask, 3] * r
        boxes[:, 2:] /= pixel_std 
        boxes[:, 2:] *= 1.25
        return boxes

    def predict_poses(self, boxes, img):
        image_patches = []
        for cx, cy, w, h in boxes:
            trans = get_affine_transform(np.array([cx, cy]), np.array([w, h]), self.patch_size)
            img_patch = cv2.warpAffine(img, trans, self.patch_size, flags=cv2.INTER_LINEAR)
            img_patch = self.pose_transform(img_patch)
            image_patches.append(img_patch)

        image_patches = torch.stack(image_patches).to(self.device)

        if not use_tflite:
            pred=self.pose_model(image_patches)
        else:
            preds=[]
            for i in range(image_patches.shape[0]):
                im=image_patches[i,:,:,:] #.permute(0,2,1)
                self.interpreter2.set_tensor(self.input_details2[0]['index'], im.unsqueeze(0))
                self.interpreter2.invoke()
                # The function `get_tensor()` returns a copy of the tensor data.
                # Use `tensor()` in order to get a pointer to the tensor.
                output_data = self.interpreter2.get_tensor(self.output_details2[0]['index'])
                preds.append(output_data)
            pred=torch.tensor(preds)
            pred=pred[:,0,:,:,:]#.permute(0,1,3,2)
            print(pred.shape)
        #print('pose out', pred.shape) #pose out torch.Size([2, 17, 64, 48])
        return pred

    def postprocess(self, pred, img1, img0):
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=0)

        for det in pred:
            if len(det):
                boxes = scale_boxes(det[:, :4], img0.shape[:2], img1.shape[-2:]).cpu()
                boxes = self.box_to_center_scale(boxes)
                outputs = self.predict_poses(boxes, img0)

                if 'simdr' in self.model_name:
                    coords = get_simdr_final_preds(*outputs, boxes, self.patch_size)
                else:
                    coords = get_final_preds(outputs, boxes)

                draw_keypoints(img0, coords, self.coco_skeletons)
                

    @torch.no_grad()
    def predict(self, image):
        
        if not use_tflite:
            img = self.preprocess(image)
            pred = self.det_model(img)[0]  
        else:
            img = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img=torch.tensor(img).permute(2,0,1).unsqueeze(0)
            img1=img/255.0
            
            print(img1.shape)
            self.interpreter.set_tensor(self.input_details[0]['index'], img1)

            self.interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = self.interpreter.get_tensor(self.output_details[3]['index'])
            pred=torch.tensor(output_data)
            print(pred.shape)

        self.postprocess(pred, img, image)
        return image


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='assests/test.jpg')
    parser.add_argument('--det-model', type=str, default='crowdhuman_yolov5m.pt')
    parser.add_argument('--pose-model', type=str, default='posehrnet_w32_256x192.pth')
    parser.add_argument('--img-size', type=int, default=320)
    parser.add_argument('--conf-thres', type=float, default=0.4)
    parser.add_argument('--iou-thres', type=float, default=0.1)
    return parser.parse_args()


if __name__ == '__main__':
    setup_cudnn()
    args = argument_parser()
    pose = Pose(
        args.det_model,
        args.pose_model,
        args.img_size,
        args.conf_thres,
        args.iou_thres
    )

    source = Path(args.source)

    if source.is_file() and source.suffix in ['.jpg', '.png']:
        image = cv2.imread(str(source))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = pose.predict(image)
        cv2.imwrite(f"{str(source).rsplit('.', maxsplit=1)[0]}_out.jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    elif source.is_dir():
        files = source.glob("*.png")
        for file in files:
            image = cv2.imread(str(file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output = pose.predict(image)
            cv2.imwrite(f"{str(file).rsplit('.', maxsplit=1)[0]}_out.jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    elif source.is_file() and source.suffix in ['.mp4', '.avi']:
        reader = VideoReader(args.source)
        writer = VideoWriter(f"{args.source.rsplit('.', maxsplit=1)[0]}_out.mp4", reader.fps)
        fps = FPS(len(reader.frames))
        print(fps)
        for frame in tqdm(reader):
            fps.start()
            output = pose.predict(frame.numpy())
            fps.stop(False)
            writer.update(output)
            # break
        
        print(f"FPS: {fps.fps}")
        writer.write()

    else:
        webcam = WebcamStream()
        fps = FPS()

        for frame in webcam:
            fps.start()
            output = pose.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fps.stop()
            cv2.imshow('frame', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
