import cv2 
import numpy as np 
import cv2 as cv


mask=cv2.imread('./data/mask.png', cv2.IMREAD_GRAYSCALE)

ret,thresh = cv2.threshold(mask, 40, 255, 0)
try:
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
except:
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

lens=[len(c) for c in contours]
pool_contour_tight=contours[lens.index(max(lens))]

mask=np.zeros(mask.shape, dtype=np.uint8)
mask=cv2.drawContours(mask, pool_contour_tight, -1, (255),cv2.FILLED)


img_dilation = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=90) 


try:
    contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
except:
    im2, contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

lens=[len(c) for c in contours]
pool_contour_outer=contours[lens.index(max(lens))]

print(pool_contour_outer.shape)

# cv2.imshow('sample', img_dilation)


# cv2.imshow('sample', mask)
# cv2.waitKey(0)

# dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
# print(dist.shape)

# ring = cv2.inRange(dist, 19.5, 20.5) # take all pixels at distance between 9.5px and 10.5px

# cv2.imshow('sample', ring)
# cv2.waitKey(0)


# contours, hierarchy = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



# # mu=np.mean(pool_contour_tight,axis=0,keepdims=True)

# # pool_zero=1.2*(pool_contour_tight-mu)+mu
# # pool_zero=pool_zero.astype('int32')

frame=cv2.imread('/data2/Freelancer/PoolAngel/Data/PoolAngel2/vid2-008.jpg')

print(pool_contour_tight.shape, pool_contour_tight.dtype)

cv2.drawContours(frame, pool_contour_tight[:,np.newaxis,:], -1, (0, 0, 255), 3)

cv2.drawContours(frame, pool_contour_outer[:,np.newaxis,:], -1, (30, 255, 255), 3)
# cv2.drawContours(frame, contours, 1, (255,0,0), 3)

# print(pool_zero.shape)

# cv2.drawContours(frame, pool_zero[:,np.newaxis,:], -1, (0, 255, 0), 3)

cv2.imshow('sample', frame)
cv2.waitKey(0)