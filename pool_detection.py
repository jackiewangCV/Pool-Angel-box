import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import subprocess

def rle2mask(mask_rle: str, label=1, shape=(1520,2688)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)  # Needed to align to RLE direction


# url='http://127.0.0.1:8000/'



def get_pool(image, url):

    if not isinstance(image,str):
        cv2.imwrite('./data/sample_image.png', image)
        image='./data/sample_image.png'
    mask=[]
    
    if not os.path.isfile('./data/mask.png'):
        if url=='local':
            print('Using local mobile SAM')
            subprocess.run(["python", "segmentation/mobileSAM_onnx.py", image])  ## subprpcess is to force the pool model to be removed from RAM, otherwise we need to depend on the garbage collection
            mask=cv2.imread('./data/mask.png', cv2.IMREAD_GRAYSCALE)
        else:
            file={'file':open(image,'rb')}

            resp=requests.post(url+'detect_pool', files=file)
            resp=resp.json()['message']

            mask=rle2mask(resp['rle'],shape=resp['size'])
            mask=mask*255
            cv2.imwrite('./data/mask.png', mask)
    else:
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


    img_dilation = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=30)  ## 30 is the correct value..!


    try:
        contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except:
        im2, contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    lens=[len(c) for c in contours]
    pool_contour_outer=contours[lens.index(max(lens))]

    return mask, pool_contour_tight, pool_contour_outer



if __name__ == '__main__':
    image='/data2/Freelancer/PoolAngel/Data/PoolAngel2/vid2-001.jpg'
    im=cv2.imread(image)
    mask,pool_contour=get_pool(image)
    cv2.drawContours(im, pool_contour, -1, (0, 0, 255), 3)
    plt.imshow(im[:,:,::-1])
    plt.show()
