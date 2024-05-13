import cv2
from sources import (Source, BenchMark)
from pool_detection import get_pool
import numpy as np
from datetime import datetime
import time

import os
import sys
sys.path.append("new_code")
from utils.trt_people_det import Yolov8_det_TRT
from utils.trt_people_cls import Yolov8_cls_TRT
from tracker.track_engine import TrackerInterface
from property import Person

class PoolAngel:
    def __init__(self, source) -> None:
        self.source = Source(source)
        
        self.input_name = source.split("/")[-1]
        self.server_key = "YXV0aG9yIGh0dHBzOi8vZ2l0aHViLmNvbS9sZWRpbmh0cmk5NyBQb29sIHByb2plY3QgZnJlZWxhbmNl"
        self.people_det = Yolov8_det_TRT("new_code/models/yolov8s-640.onnx.engine")
        self.people_cls = Yolov8_cls_TRT("new_code/models/child-adult-cls-yolov8s-128.onnx.engine")
        self.tracker =  TrackerInterface()
        self.bechmark = BenchMark()
        self.pool_contour = None
        self.zone_det = None
        self.video = None
        self.url = "local"
        
        output_dir = "./data/output"
        os.makedirs(output_dir, exist_ok=True)
        self.output_video = f"{output_dir}/output_{self.input_name}.mkv"
        self.out_width = 640
        self.out_height = 480
        input_fps = 25
        self.video_writer = cv2.VideoWriter(self.output_video, 
                cv2.VideoWriter_fourcc(*"MJPG"), input_fps, (self.out_width, self.out_height))

        self.pool_width = 0
        self.max_width = 0
        
    def run(self):
                    
        self.bechmark.start("Whole process")
        frame_c = 0
        frame_skip = 2
        start_t = time.time()
        
        while True:
            # self.bechmark.start("Frame procesisng")

            ### Frame capture
            # self.bechmark.start("Frame Capture")
            has_frame, frame = self.source.get_frame()
            # self.bechmark.stop("Frame Capture")

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
                x, _, width, height = cv2.boundingRect(self.pool_contour_outer)
                xmin = max(0, x - width // 5)
                xmax = min(max_width, x + width + width // 5)
                self.zone_det = (xmin, 0, xmax, max_height)
                print(self.mask.shape)
                self.bechmark.stop("mask detection")
                self.pool_width = width
                self.max_width = max_width

            ### Object detection 
            self.bechmark.start("Object detection")
            bboxes, scores, labels = self.people_det.predict(frame, zone_det=self.zone_det)
            track_ids = self.tracker.track(bboxes, scores)
            persons = []
            colors = []
            typs = []
            list_position_aware = []
            list_position_adult = []
            for oid, bb, s, lb in zip(track_ids, bboxes, scores, labels):
                
                p_height = bb[3] - bb[1]
                
                ps = Person(oid, bbox=bb, confidence=s)
                bb = list(map(int, bb))
                crop_people = frame[bb[1]:bb[3], bb[0]:bb[2]]
                label_cls, conf_cls = self.people_cls.predict(crop_people)
                if label_cls == 1:
                    tp = f"child {conf_cls:.2f}"
                else:
                    tp = f"adult {conf_cls:.2f}"
                
                # ps.detect_slip()
                ps.update_distance(self.pool_contour, self.pool_contour_outer)
                persons.append(ps)
                is_aware = True
                if ps.dist_pool == 0:
                    clr = (0, 0, 255) 
                    # people is dangerous please send SOS
                elif ps.warn:
                    clr = (30, 255, 255) 
                    # people is quite close to the pool please send warning
                else:
                    clr = (0, 255, 0)
                    is_aware = False
                    # no event people is safe
                colors.append(clr)
                
                if "adult" in tp:
                    list_position_adult.append((oid, bb))
                elif "child" in tp and is_aware:
                    list_position_aware.append((oid, bb))
                typs.append(f"{tp} {p_height:.1f}")
                print(f"[{frame_c}] {s[0]:.2f} Pose id {oid}: typ {tp} rat {p_height:.2f}")
            self.bechmark.stop("Object detection")
            
            for oid, bb in list_position_aware:
                offset_alert = 20
                xmin = int(bb[0] - offset_alert)
                ymin = int(bb[1] - offset_alert)
                xmax = int(bb[2] + offset_alert)
                ymax = int(bb[3] + offset_alert)
                message_topic1 = f"alert topic1: child {oid}"
                if len(list_position_adult) == 0:
                    print(message_topic1)
                    cv2.putText(frame, message_topic1, (xmin, ymax), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2, cv2.LINE_AA)
                else:
                    min_dis_from_child_to_adult = self.max_width
                    for (_, bb2) in list_position_adult:
                        dis_pp = np.sqrt((bb[0]-bb2[0])**2 + (bb[1]-bb2[1])**2)
                        if min_dis_from_child_to_adult > dis_pp:
                            min_dis_from_child_to_adult = dis_pp
                    if min_dis_from_child_to_adult > self.pool_width / 4:
                        print(message_topic1)
                        cv2.putText(frame, message_topic1, (xmin, ymax), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2, cv2.LINE_AA)
                
            ## visualize detections
            # self.bechmark.start("Visualization")
            cv2.drawContours(frame, pool_contour[:,np.newaxis,:], -1, (0, 0, 255), 3)
            cv2.drawContours(frame, pool_contour_outer[:,np.newaxis,:], -1, (30, 255, 255), 3)
            cv2.putText(frame, f"Frame {frame_c}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0, 0, 255), thickness=3)
            cv2.rectangle(frame, (self.zone_det[0], self.zone_det[1]), 
                        (self.zone_det[2], self.zone_det[3]), (255, 0, 255), 3, cv2.LINE_AA)
            vis_img = self.people_det.vis_people(frame, bboxes, scores, 
                    self.out_width, self.out_height, 
                    track_ids=track_ids, typs=typs, colors=colors)  ## Last argment is hardcoded 
            self.video_writer.write(vis_img)
            # cv2.imwrite("t.jpg", vis_img)
            # self.bechmark.stop("Visualization")

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
        self.video_writer.release()
        print(f"Save video {self.output_video}")
        total_t = time.time() - start_t
        print(f"{frame_c} in {total_t} -> FPS: {frame_c / total_t}")
        self.people_det.destroy()
        self.people_cls.destroy()