from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import cv2
from .utils import mask2rle


class SAM_pooldetection:
    def __init__(self,model_type, checpoint_path, ) -> None:
        self.sam = sam_model_registry[model_type](checkpoint=checpoint_path)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
    
    def segment(self, image):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
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
        for mask_data in masks:
            mask = mask_data["segmentation"]
            olp= np.sum(mask_in*mask)
            if olp>overlap:
                final_mask=mask.copy()
                print('detected one of the mask')
                overlap=olp
                np.save(f'./data/mask_{cnt}',final_mask)
                cnt+=1
                #break
        return final_mask


