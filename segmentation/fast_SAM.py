import numpy as np
import cv2
from .utils import mask2rle
from fastsam import FastSAM, FastSAMPrompt
import matplotlib.pyplot as plt


DEVICE = 'cpu'


class fastSAM_pooldetection:
    def __init__(self) -> None:
        self.model = FastSAM('./data/FastSAM-x.pt')
    
    def segment(self, IMAGE_PATH):
        image = cv2.imread(IMAGE_PATH)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        everything_results = self.model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
        prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)
        # everything prompt
        masks = prompt_process.everything_prompt()

        mask=self.select_mask(masks, image)
        rle=mask2rle(mask)
        return mask, rle, mask.shape
    
    def select_mask(self, masks, image):
        lower_range= np.array([180//2,50,50])
        upper_range= np.array([210//2,255,255])
        mask_in =cv2.inRange(image, lower_range, upper_range)
        mask=[]
        overlap=0.0
        cnt=1
        masks=masks.numpy()>0
        
        for i in range(masks.shape[0]):
            mask = masks[i,:,:]
            olp= np.sum(mask_in*mask)

            if olp>overlap:
                final_mask=mask.copy()
                
                print('detected one of the mask')
                overlap=olp
                # np.save(f'./data/mask_{cnt}',final_mask)
                cnt+=1
                #break
        return final_mask


