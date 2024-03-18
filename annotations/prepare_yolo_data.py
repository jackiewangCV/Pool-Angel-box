import glob
import shutil
import numpy as np
import cv2


img_path='/data1/ImageData/CrowdHuman/Images/'

data_dir='/data2/Freelancer/PoolAngel/Data/crowdHumanFiltered/'


labels=data_dir+'labels/'

labels_yolo=data_dir+'labels_yolo/'


lab_files=glob.glob(labels+'*.txt')
lab_files.sort()


for lab in lab_files:
    filename=lab.split('/')[-1][:-4]
    im=cv2.imread(img_path+filename+'.jpg')

    height, width, _=im.shape
    input = np.loadtxt(lab, dtype='i', delimiter=' ')
    input=input.astype('float')

    #label (0),xmin (1), ymin (2), w (3),h (4)

    input[:,1]=input[:,1]+input[:,3]/2
    input[:,2]=input[:,2]+input[:,4]/2

    input[:,1]=input[:,1]/width
    input[:,2]=input[:,2]/height

    input[:,3]=input[:,3]/width
    input[:,4]=input[:,4]/height

    np.savetxt(labels_yolo+filename+'.txt',input, fmt='%i %1.6f %1.6f %1.6f %1.6f')
    cv2.imwrite(data_dir+'Images/'+filename+'.jpg', im)
    # break



