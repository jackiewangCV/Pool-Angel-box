from segmentation.fast_SAM import fastSAM_pooldetection
import matplotlib.pyplot as plt

IMAGE_PATH = '/data2/Freelancer/PoolAngel/Data/PoolAngel2/vid2-001.jpg'

detector=fastSAM_pooldetection()

mask,_,_=detector.segment(IMAGE_PATH)
print(mask.shape)

plt.imshow(mask)
plt.show()