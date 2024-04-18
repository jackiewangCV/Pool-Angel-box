import cv2
from sources import (Source, BenchMark)
from pool_detection import get_pool
import numpy as np
from datetime import datetime
import time

import os
import sys
sys.path.append("new_code")
from utils.trt_pose import Pose_TRT
from property import Person

class PoolAngel:
    def __init__(self, source) -> None:
        self.source = Source(source)
        
        self.input_name = source.split("/")[-1]
        self.pose_model = Pose_TRT("new_code/models/yolov8s-pose-640.onnx.engine", img_size=640)
        self.bechmark = BenchMark()
        self.pool_contour = None
        self.zone_det = None
        self.video = None
        self.url = "local"
        
        output_dir = "./data/output"
        os.makedirs(output_dir, exist_ok=True)
        output_video = f"{output_dir}/output_{self.input_name}.mkv"
        self.out_width = 640
        self.out_height = 480
        input_fps = 25
        self.video_writer = cv2.VideoWriter(output_video, 
                cv2.VideoWriter_fourcc(*"MJPG"), input_fps, (self.out_width, self.out_height))

        self.num_split_y = 2
        self.range_y_axis_height = []
        
    def run(self):
                    
        self.bechmark.start("Whole process")
        frame_c = 0
        frame_skip = 2
        start_t = time.time()
        
        while True:
            # self.bechmark.start("Frame procesisng")

            ### Frame capture
            self.bechmark.start("Frame Capture")
            has_frame, frame = self.source.get_frame()
            self.bechmark.stop("Frame Capture")

            if not has_frame:
                print("End of video")
                if self.video is not None:
                    self.video.release()
                break
            frame_c += 1
            if frame_c % frame_skip != 0:
                continue

            if self.pool_contour is None:
                print("Doing pool detection for first time..")
                self.bechmark.start("mask detection")
                self.mask, pool_contour, pool_contour_outer = get_pool(self.input_name, frame, self.url)
                self.pool_contour = pool_contour[:,0,:]
                self.pool_contour_outer = pool_contour_outer[:,0,:]
                
                max_height, max_width = frame.shape[:2]
                x, _, width, _ = cv2.boundingRect(self.pool_contour_outer)
                xmin = max(0, x - width // 4)
                xmax = min(max_width, x + width + width // 4)
                self.zone_det = (xmin, 0, xmax, max_height)
                print(self.mask.shape)
                self.bechmark.stop("mask detection")
                
                range_y = 0
                length_y = max_height // self.num_split_y
                for i in range(self.num_split_y):
                    self.range_y_axis_height.append({
                        "ymin": range_y,
                        "ymax": (range_y + length_y),
                        "max_p_height": None,
                        "min_p_height": None
                    })
                    range_y += length_y

            ### Object detection 
            self.bechmark.start("Object detection")
            bboxes, scores, kpts = self.pose_model.predict(frame, self.zone_det)
            persons = []
            colors = []
            typs = []
            for i, bb in enumerate(bboxes):
                rat, p_height = self.pose_model.compute_rat(kpts[i])
                
                ps = Person(i, bbox=bb, pose=kpts[i], confidence=scores[i], ratio=rat)
                tp = ps.child_or_adult()
                
                for range_y_j in self.range_y_axis_height:                    
                    if range_y_j["ymin"] < bb[3] < range_y_j["ymax"]:
                        if range_y_j["max_p_height"] is None:
                            range_y_j["max_p_height"] = p_height
                        elif range_y_j["max_p_height"] < p_height:
                            range_y_j["max_p_height"] = p_height
                        if range_y_j["min_p_height"] is None:
                            range_y_j["min_p_height"] = p_height
                        elif range_y_j["min_p_height"] > p_height:
                            range_y_j["min_p_height"] = p_height
                        if range_y_j["max_p_height"] is not None and \
                            range_y_j["min_p_height"] is not None and \
                            range_y_j["min_p_height"] / range_y_j["max_p_height"] < 0.6:
                            rat = p_height / ((range_y_j["max_p_height"] + range_y_j["min_p_height"]) / 2)
                            if rat < 0.85:
                                tp = f"child {rat:.2f}"
                            else:
                                tp = f"adult {rat:.2f}"    
                        break
                
                ps.detect_slip()
                ps.update_distance(self.pool_contour, self.pool_contour_outer)
                persons.append(ps)
                if ps.dist_pool == 0:
                    clr = (0, 0, 255) 
                    # people is dangerous please send SOS
                elif ps.warn:
                    clr = (30, 255, 255) 
                    # people is quite close to the pool please send warning
                else:
                    clr = (0, 255, 0) 
                    # no event people is safe
                colors.append(clr)
                
                typs.append(f"{tp} {p_height:.1f}")
                print(f"[{frame_c}] {scores[i][0]:.2f} Pose {i} : typ {tp} rat {p_height:.2f}")
            self.bechmark.stop("Object detection")
            
            ## visualize detections
            self.bechmark.start("Visualization")
            cv2.drawContours(frame, pool_contour[:,np.newaxis,:], -1, (0, 0, 255), 3)
            cv2.drawContours(frame, pool_contour_outer[:,np.newaxis,:], -1, (30, 255, 255), 3)
            cv2.putText(frame, f"Frame {frame_c}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0, 0, 255), thickness=3)
            cv2.rectangle(frame, (self.zone_det[0], self.zone_det[1]), 
                        (self.zone_det[2], self.zone_det[3]), (255, 0, 255), 3, cv2.LINE_AA)
            vis_img = self.pose_model.vis_pose(frame, bboxes, scores, kpts, 
                    self.out_width, self.out_height, typs=typs, colors=colors)  ## Last argment is hardcoded
            self.video_writer.write(vis_img)
            # cv2.imwrite("t.jpg", vis_img)
            self.bechmark.stop("Visualization")

            # time_now=datetime.now()
            # if (time_now-last_img_store).total_seconds()>100:  ## store only if last event has happened in last 10min
            #     ## recording
            #     for per in persons:
            #         if per.dist_pool == 0 or per.warn or per.is_slipped:
            #             print("saving image")
            #             last_img_store = time_now
            #             cv2.imwrite("./data/"+str(time_now)+".png", vis_img)
            #             break
            
            # self.bechmark.stop("Frame procesisng")
        self.bechmark.stop("Whole process")
        total_t = time.time() - start_t
        print(f"{frame_c} in {total_t} -> FPS: {frame_c / total_t}")
        self.pose_model.destroy()