import cv2
import glob
import numpy as np

# name='vid2-005'
name='img3_0002'

image = cv2.imread(f'data/images/{name}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


masks=glob.glob(f'data/masks_{name}/*.png')

for mm in masks:
    mask=cv2.imread(mm)
    perc=np.sum(mask[:,:,0]==255)/np.prod(mask.shape[:2])
    
    color_h=image[mask[:,:,0]==255,0].mean()
    color_s=image[mask[:,:,0]==255,1].mean()
    color_v=image[mask[:,:,0]==255,2].mean()
    
    if perc>0.04:
        print(image.shape)
        print(mm, perc, np.prod(mask.shape[:2]), mask.shape, color_h, color_s, color_v)

