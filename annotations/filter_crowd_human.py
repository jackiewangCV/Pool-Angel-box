import glob
import sys
import matplotlib.pyplot as plt
import cv2 
import shutil
import numpy as np
import json
import time


def click_event(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX.append(int(x))
        mouseY.append(int(y))
        print('selected bbox')
    if event == cv2.EVENT_RBUTTONDBLCLK:
        mouseX,mouseY = 0,0



base_path='/data1/ImageData/CrowdHuman/'

ann_path=base_path+'annotation_train.odgt'


f=open(ann_path);ann_data=f.readlines();f.close()

out_path='/data2/Freelancer/PoolAngel/Data/crowdHumanFiltered/labels_annotated/'

ann_files=glob.glob(out_path+'*.txt')
ann_files=[f.split('/')[-1] for f in ann_files]
# img = np.zeros((512,512,3), np.uint8)
# cv2.namedWindow('image')


# global mouseX,mouseY
# mouseX=''
## finshed till 922
num=971

ann_data=ann_data[num:]
for inds,d in enumerate(ann_data):
    data=json.loads(d)
    print(inds+num, data['ID'])

    if data['ID']+'.jpg' in ann_files:
        print('ann exists')
        continue

    im=cv2.imread(base_path+'Images/'+data['ID']+'.jpg')
    boxes=data['gtboxes']
    mouseX,mouseY =[] , []

    if len(boxes)>5:
        continue

    for box in boxes:
        if box['tag']=='person':
            xmin, ymin, w,h=box['hbox']
            xmax=xmin+w
            ymax=ymin+h
            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (255,0,0), thickness=2)

    cv2.imshow('image', im)
    cv2.setMouseCallback('image', click_event) 
    cv2.waitKey(0)


    if (mouseX==[]):
        print('skipping')
    else:
        f=open(out_path+data['ID']+'.txt','w')

        im=cv2.imread(base_path+'Images/'+data['ID']+'.jpg')
        print(mouseX)
        for box in boxes:
            if box['tag']=='person':
                xmin, ymin, w,h=box['hbox']
                xmax=xmin+w
                ymax=ymin+h
                selected=False
                for mx,my in zip(mouseX,mouseY):
                    if (mx>xmin and mx<xmax and my>ymin and my<ymax):
                       selected=True
                       break
                    
                if selected:
                    cls='1'
                    cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0,255,0), thickness=2)
                else:
                    cls='0'
                    cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (255,0,0), thickness=2)

                bb=box['vbox']
                bb=[str(b) for b in bb]
                line=cls+' '+' '.join(bb)+'\n'
                print(line)
                f.write(line)
        
        f.close()

        cv2.imshow('image', im)
        cv2.setMouseCallback('image', click_event) 
        cv2.waitKey(0)

    # break




# img_bgr=cv2.imread(f)
# cv2.imshow(f,img_bgr)
# cv2.waitKey(0)



# def draw_circle(event,x,y,flags,param):
#     global mouseX,mouseY
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img_bgr,(x,y),100,(255,0,0),-1)
#         mouseX,mouseY = x,y

# files.sort()

# print(files)

# img = np.zeros((512,512,3), np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)

# for i,f in enumerate(files):
#     img_bgr=cv2.imread(f)
#     print(img_bgr.shape)

#     # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     while(1):
#         cv2.imshow('image',img_bgr)
#         k = cv2.waitKey(20) & 0xFF
#         if k == 27:
#             break
#         elif k == ord('a'):
#             print(mouseX,mouseY)

#     break
