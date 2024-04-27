from tracker.sort.sort import Sort
import numpy as np

class TrackerInterface(object):
    
    def __init__(self):
        self.tracker = Sort(max_age=20)
        
    def track(self, bboxes, scores):
        detection_list = np.empty((0, 5))
        for bb, s in zip(bboxes, scores):
            x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]
            conf = s[0]
            detection_list = np.vstack((detection_list, np.array([x1, y1, x2, y2, conf])))
        track_bbs_ids = self.tracker.update(detection_list)
        track_ids = []
        for bb in bboxes:
            for dd in track_bbs_ids:
                is_match = True
                for i in range(4):
                    if bb[i] - dd[i] != 0:
                        is_match = False
                        break
                if is_match:
                    track_ids.append(dd[4])
                    break
        print("track_ids:", track_ids)
        return track_ids
        
        