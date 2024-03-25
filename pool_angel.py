import cv2
from sources import Source
from object_detection import ObjectDetection
from pose_detection import PoseDetection
from object_tracking import ObjectTracking
from pool_detection import get_pool
import numpy as np
import time
import utils
import cv2 as cv
from datetime import datetime

DETECT_EVERY_N_FRAMES=0

class PoolAngel:
    def __init__(self, source, detector='nanodet', tracker=None, posenet='pose_detector', visualize=False, save=False, server_url=None) -> None:
        self.source=Source(source)
        self.det_name=detector
        self.detector=ObjectDetection(detector, None)
        self.pose_detection=PoseDetection(posenet)
        self.bechmark=utils.BenchMark()
        self.visualize=visualize
        self.pool_contour=None
        self.video=None
        self.save=save
        self.url=server_url
        self.tracker=None
        if tracker is not None:
            self.tracker=ObjectTracking(tracker=tracker)
        

        if self.tracker is not None:
            self.track_frame=DETECT_EVERY_N_FRAMES
        else:
            self.track_frame=0


    def run(self):
        
        if not self.save:
            cv.namedWindow('Demo', cv2.WINDOW_NORMAL)
                    
        self.bechmark.start("Whole process")
        last_img_store=datetime(2013,12,30,23,59,59)

        while(cv.waitKey(1) < 0):
            self.bechmark.start("Frame procesisng")

            ### Frame capture
            self.bechmark.start("Frame Capture")
            has_frame, frame=self.source.get_frame()
            self.bechmark.stop("Frame Capture")

            if not has_frame:
                print('End of video')
                if self.video is not None:
                    self.video.release()
                break
            

            
            if self.pool_contour is None:
                print('Doing pool detection for first time..')
                self.bechmark.start("mask detection")
                self.mask,pool_contour, pool_contour_outer=get_pool(frame, self.url)
                self.pool_contour=pool_contour[:,0,:]
                self.pool_contour_outer=pool_contour_outer[:,0,:]
                print(self.mask.shape)
                self.bechmark.stop("mask detection")

            self.track_frame=self.track_frame+1

            if self.track_frame>DETECT_EVERY_N_FRAMES:
                self.track_frame=0
                ### Object detection 
                self.bechmark.start("Object detection")
                persons=self.detector.detect(frame)

                for i, per in enumerate(persons):
                    persons[i].update_distance(self.pool_contour, self.pool_contour_outer)

                self.bechmark.stop("Object detection")

                ## If object is detected
                if len(persons)>0:
                    self.bechmark.start("Pose detection")
                    ## Do pose detection 
                    persons=self.pose_detection.detect(frame, persons)
                    print("Num person:", len(persons))
                    self.bechmark.stop("Pose detection")
                    
            else:
                if len(persons)==0:
                    self.track_frame=DETECT_EVERY_N_FRAMES
                
                self.bechmark.start("Object tracking")
                ## object tracking...! 
                if self.tracker is not None:
                    for i, per in enumerate(persons):
                        bbox=per.bbox[-1]
                        bbox=[round(bbox[0]),round(bbox[1]),round(bbox[2]-bbox[0]),round(bbox[3]-bbox[1])]
                        print(bbox)
                        try:
                            self.tracker.init(last_frame, bbox)
                            is_located, bbox, confidence=self.tracker.track(frame)
                            if is_located and confidence>0.6:
                                bbox=[bbox[0],bbox[1],bbox[2]+bbox[0],bbox[3]+bbox[1]]
                                persons[i].update_infos(bbox=bbox)
                            else:
                                print('Object is gone')
                                self.track_frame=DETECT_EVERY_N_FRAMES
                        except Exception as e:
                            print(e)
                            print(last_frame, bbox)
                self.bechmark.stop("Object tracking")

            last_frame=frame

            if self.visualize:
                ## visualize detections
                self.bechmark.start("Visualization")
                cv2.drawContours(frame, pool_contour[:,np.newaxis,:], -1, (0, 0, 255), 3)
                cv2.drawContours(frame, pool_contour_outer[:,np.newaxis,:], -1, (30,255, 255), 3)

                if self.det_name=='yolox':
                    scale=(139, 0, 361, 640)
                elif self.det_name=='yolov8':
                    scale=None
                else:
                    scale=(90, 0, 235, 416)

                img = utils.vis(persons, frame, scale)  ## Last argment is hardcoded
                if not self.save:
                    cv.imshow('Demo', img)
                    # cv2.waitKey(0)
                else:
                    if self.video is None:
                        im_size=frame.shape[:2]
                        self.video= cv2.VideoWriter(self.save, cv2.VideoWriter_fourcc('F','M','P','4'), 30, (im_size[1],im_size[0]))

                    print(img.shape,im_size) 
                    self.video.write(img) 


                self.bechmark.stop("Visualization")

            time_now=datetime.now()
            # print((time_now-last_img_store).total_seconds())
            if (time_now-last_img_store).total_seconds()>100:  ## store only if last event has happened in last 10min
                ## recording
                for per in persons:
                    if per.dist_pool==0 or per.warn or per.is_slipped:
                        print('saving image')
                        last_img_store=time_now
                        cv2.imwrite('./data/'+str(time_now)+'.png',frame)
                        break
            
            self.bechmark.stop("Frame procesisng")
        self.bechmark.stop("Whole process")