import cv2
import time

class BenchMark:
    def __init__(self) -> None:
        self.tasks={}
    
    def start(self, name):
        self.tasks[name]=time.time()

    def stop(self, name):
        tm=time.time()-self.tasks[name]
        print(f'{name} took {tm} seconds')
        return tm
    
class Source:
    def __init__(self,source_type) -> None:
        self.has_frame=True
        if source_type[-4:] in ['.mp4'] or 'rtsp' in source_type:
            self.video = cv2.VideoCapture(source_type)
            self.type=0
        elif source_type in ['png','jpg','jpeg']:
            self.first_frame=cv2.imread(source_type)
            self.type=1
        else:
            NotImplemented("Other video sources are not implemented yet..!")
            
    def get_frame(self):
        if self.type==0:
            has_frame, first_frame = self.video.read()
        elif self.type==1:
            has_frame=self.has_frame 
            first_frame=self.first_frame
            self.has_frame=False
        return has_frame, first_frame
