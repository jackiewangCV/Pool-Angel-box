import numpy as np
import cv2
# from segmentation.utils import mask2rle
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

DEVICE = 'cpu'

def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

class MobileSAM_pooldetection:
    def __init__(self) -> None:
        model_type = "vit_t"
        sam_checkpoint = "./data/mobile_sam.pt"

        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.to(device=DEVICE)
        mobile_sam.eval()

        self.mask_generator = SamAutomaticMaskGenerator(mobile_sam,points_per_side=4,points_per_batch=1)
    
    def segment(self, IMAGE_PATH):
        image = cv2.imread(IMAGE_PATH)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # everything prompt

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
        
        for i in range(len(masks)):
            mask = masks[i]['segmentation']
            olp= np.sum(mask_in*mask)

            if olp>overlap:
                final_mask=mask.copy()
                
                print('detected one of the mask')
                overlap=olp
                # np.save(f'./data/mask_{cnt}',final_mask)
                cnt+=1
                #break
        return final_mask


import sys
if __name__ == "__main__": 
    image_path=sys.argv[1]
    model=MobileSAM_pooldetection()
    mask,_,_=model.segment(image_path)
    mask=np.uint8(mask*255)
    cv2.imwrite('./data/mask.png', mask)

