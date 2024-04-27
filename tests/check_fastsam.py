from segmentation.fast_SAM import fastSAM_pooldetection
IMAGE_PATH = '/data2/Freelancer/PoolAngel/Data/PoolAngel2/vid2-001.jpg'

detector=fastSAM_pooldetection()

mask,_,_=detector.segment(IMAGE_PATH)
print(mask.shape)