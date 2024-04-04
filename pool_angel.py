import cv2
from sources import (Source, BenchMark)
from pool_detection import get_pool
import numpy as np
from datetime import datetime

import os
from new_code.utils.trt_pose import Pose_TRT
from property import Person

class PoolAngel:
    def __init__(self, source, server_url=None) -> None:
        self.source = Source(source)

        self.pose_model = Pose_TRT("new_code/models/yolov8n-pose.onnx.engine")
        self.bechmark = BenchMark()
        self.pool_contour = None
        self.video = None
        self.url = server_url
        
        output_dir = "./data/output"
        os.makedirs(output_dir, exist_ok=True)
        output_video = f"{output_dir}/output.mkv"
        out_width = 640
        out_height = 480
        input_fps = 25
        self.video_writer = cv2.VideoWriter(output_video, 
                cv2.VideoWriter_fourcc(*"MJPG"), input_fps, (out_width, out_height))

    def run(self):
                    
        self.bechmark.start("Whole process")
        last_img_store = datetime(2013,12,30,23,59,59)

        while True:
            self.bechmark.start("Frame procesisng")

            ### Frame capture
            self.bechmark.start("Frame Capture")
            has_frame, frame = self.source.get_frame()
            self.bechmark.stop("Frame Capture")

            if not has_frame:
                print("End of video")
                if self.video is not None:
                    self.video.release()
                break
            
            if self.pool_contour is None:
                print("Doing pool detection for first time..")
                self.bechmark.start("mask detection")
                self.mask,pool_contour, pool_contour_outer=get_pool(frame, self.url)
                self.pool_contour=pool_contour[:,0,:]
                self.pool_contour_outer=pool_contour_outer[:,0,:]
                print(self.mask.shape)
                self.bechmark.stop("mask detection")

            ### Object detection 
            self.bechmark.start("Object detection")
            bboxes, scores, kpts = self.pose_model.predict(frame)
            persons=[]
            for i, bb in enumerate(bboxes):
                rat = self.pose_model.compute_rat(kpts[i])
                ps = Person(i, bbox=bboxes[i], pose=kpts[i], confidence=scores[i], ratio=rat)
                ps.child_or_adult()
                ps.detect_slip()
                
                ps.update_distance(self.pool_contour, self.pool_contour_outer)
                persons.append(ps)
                
            self.bechmark.stop("Object detection")


            ## visualize detections
            self.bechmark.start("Visualization")
            cv2.drawContours(frame, pool_contour[:,np.newaxis,:], -1, (0, 0, 255), 3)
            cv2.drawContours(frame, pool_contour_outer[:,np.newaxis,:], -1, (30,255, 255), 3)
            vis_img = self.pose_model.vis_pose(frame, bboxes, scores, kpts)  ## Last argment is hardcoded
            self.video_writer.write(vis_img)
            self.bechmark.stop("Visualization")

            time_now=datetime.now()
            if (time_now-last_img_store).total_seconds()>100:  ## store only if last event has happened in last 10min
                ## recording
                for per in persons:
                    if per.dist_pool == 0 or per.warn or per.is_slipped:
                        print("saving image")
                        last_img_store = time_now
                        cv2.imwrite("./data/"+str(time_now)+".png", vis_img)
                        break
            
            self.bechmark.stop("Frame procesisng")
        self.bechmark.stop("Whole process")