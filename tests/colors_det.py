import cv2
import numpy as np

# name='vid2-005'
name='img3_0002'



image = cv2.imread(f'data/images/{name}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# lower_range = np.array([110,50,50])
# upper_range = np.array([130,255,255])
print(image.max(), image.min())
lower_range= np.array([180//2,50,50])
upper_range= np.array([220//2,255,255])

mask_in =cv2.inRange(image, lower_range, upper_range)

cv2.imwrite(f'data/{name}.png', mask_in)