import requests
import os
import cv2
from segmentation.utils import rle2mask

api_url='http://127.0.0.1:8000/'

pool_detect_url=api_url+'detect_pool'


# name='vid2-005'
name='img3_0002'


test_image=f'data/images/{name}.jpg'

reponse=requests.post(url=pool_detect_url, files=[('file',open(test_image, 'rb'))])	
reponse=reponse.json()['message']

mask=rle2mask(reponse['rle'],shape=reponse['size'])

cv2.imwrite(f'data/{name}.png', mask * 255)
