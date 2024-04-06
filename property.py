import numpy as np
import cv2

class Person:
    """
       Person properties
    """
    def __init__(self, id, bbox, pose=None,dist_pool=None, confidence=0.0, ratio=0) -> None:
        self.bbox=[bbox]
        self.pose=[pose]
        self.id=id
        self.dist_pool=dist_pool
        self.confidence=confidence
        self.is_child=False
        self.is_slipped=False
        self.ratio=ratio
        self.warn=False

    def detect_slip(self):
        pose = self.pose[-1]
        pose = np.array(pose).reshape(17, 3)
        height1=np.sqrt((pose[0,1]-pose[15,1])**2+(pose[0,0]-pose[16,0])**2)
        height2=np.sqrt((pose[0,1]-pose[15,1])**2+(pose[0,0]-pose[16,0])**2)
        diff1=np.abs(pose[15,0]-pose[13,0])
        diff2=np.abs(pose[16,0]-pose[14,0])
        diff3=np.abs(pose[11,0]-pose[13,0])
        diff4=np.abs(pose[14,0]-pose[12,0])
        self.diff=np.mean(np.array([diff1, diff2, diff3, diff4]))/max(height1,height2)
        if self.diff<0.05:
            self.is_slipped=True
            print(f"slip detected {self.diff}")
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
        if self.ratio > 0.6:
            self.is_child = "child"
        else:
            self.is_child = "adult"
        return self.is_child
    
    def __str__(self):
        return "".join([str(x) for x in self.pose])
    
