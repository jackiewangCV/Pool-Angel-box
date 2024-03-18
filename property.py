import numpy as np
import cv2


MAX_FRAMES=10

class Person:
    '''
       Person properties
    '''
    def __init__(self, id, bbox, pose=None,dist_pool=None, confidence=0.0) -> None:
        self.bbox=[bbox]
        self.pose=[pose]
        self.id=id
        self.dist_pool=dist_pool
        self.confidence=confidence
        self.is_child=False
        self.is_slipped=False
        self.ratio=0
        self.warn=False

    def update_infos(self, bbox=None, poses=None, dist_pool=None, confidence=None, ratio=None):
        '''
        This updates the person features such bbox, pose and positions.
        It computes of the person is  child or if he/she has slipped based on history.
        '''

        # if len(self.bbox)>MAX_FRAMES:  ## remove history if it cross the number of frames
        #     self.bbox=self.bbox[1:]+[bbox]
        #     self.pose=self.pose[1:]+[poses]
        # else:
        #     self.bbox.append(bbox)
        #     self.pose.append(poses)
        if poses is not None:
            self.pose=poses
            self.detect_slip(poses)

        if ratio is not None:
            self.ratio=ratio
            self.is_child=self.child_or_adult()

        self.dist_pool=dist_pool
        self.confidence = confidence
        self.detect_slip(poses)
        

#KEYPOINT_DICT = {
#     'nose': 0,
#     'left_eye': 1,
#     'right_eye': 2,
#     'left_ear': 3,
#     'right_ear': 4,
#     'left_shoulder': 5,
#     'right_shoulder': 6,
#     'left_elbow': 7,
#     'right_elbow': 8,
#     'left_wrist': 9,
#     'right_wrist': 10,
#     'left_hip': 11,
#     'right_hip': 12,
#     'left_knee': 13,
#     'right_knee': 14,
#     'left_ankle': 15,
#     'right_ankle': 16
# }

    def detect_slip(self, poses):

        height1=np.sqrt((poses[0,1]-poses[15,1])**2+(poses[0,0]-poses[16,0])**2)
        height2=np.sqrt((poses[0,1]-poses[15,1])**2+(poses[0,0]-poses[16,0])**2)
        diff1=np.abs(poses[15,0]-poses[13,0])
        diff2=np.abs(poses[16,0]-poses[14,0])
        diff3=np.abs(poses[11,0]-poses[13,0])
        diff4=np.abs(poses[14,0]-poses[12,0])
        self.diff=np.mean(np.array([diff1, diff2, diff3, diff4]))/max(height1,height2)
        # print(poses)

        # print(self.diff)
        # print(self.diff, height1,height2)
        if self.diff<0.15:
            self.is_slipped=True
            print('slip detected')
        else:
            self.is_slipped=False

    def update_distance(self, pool_contour_tight, pool_contour_warning):
        bb=self.bbox[-1]
        xmin, ymin, xmax, ymax = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        dist_tight = cv2.pointPolygonTest(pool_contour_tight, (round(0.5*(xmax+xmin)), ymax), False)
        dist_warn = cv2.pointPolygonTest(pool_contour_warning, (round(0.5*(xmax+xmin)), ymax), False)
        

        if dist_tight==1.0:
            self.dist_pool=0
        else:
            self.dist_pool=dist_tight

        if dist_warn==1.0:
            self.warn=True
        else:
            self.warn=False
        
    def child_or_adult(self):
        if self.ratio>20:
            return 'child'
        return 'adult' ## Implement your algos to detect the child/adult
    
    def __str__(self):
        return ''.join([str(x) for x in self.pose])
    
