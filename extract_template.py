import cv2
import numpy as np
import glob
import string
from matplotlib import pyplot as plt
from numpy import ones,vstack
from numpy.linalg import lstsq
import time


TEMPLATE_SIZE=26
TEMPLATE_NUM=6

def good_feature_corners(img,pro_img,at):
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	quality=0.01
	maxCorner=70
	mindist=20
	mask=np.zeros(gray.shape,np.uint8)
	g=gray[mask.shape[0]//7:mask.shape[0]*4//5,mask.shape[1]//4:mask.shape[1]*6//7]
	mask[mask.shape[0]//7:mask.shape[0]*4//5,mask.shape[1]//4:mask.shape[1]*6//7]=g
	blocksize=3
	corners=cv2.goodFeaturesToTrack(gray,maxCorners=maxCorner,qualityLevel=quality,mask=mask,
		minDistance=mindist,useHarrisDetector=1,blockSize=blocksize)
	corners=np.int0(corners)
	count=0
	loc=(0,0)
	for i in corners:
		x,y=i.ravel()
		if(count==at):
			cv2.rectangle(pro_img,(x-5,y-5),(x+5,y+5),(0,0,255))
			loc=(x,y)
		cv2.circle(pro_img,(x,y),10,(0,0,255))
		count+=1
	return pro_img,loc
#17 85

def extract_template(img,pro_img,size,template_count):
	temp_count=template_count

	pro_img,loc1=good_feature_corners(img,pro_img,58)
	pro_img,loc2=good_feature_corners(img,pro_img,61)
	tmp1=np.zeros((size,size),np.uint8)
	tmp2=np.zeros((size,size),np.uint8)

	tmp1=img[loc1[1]-size//2:loc1[1]+size//2,loc1[0]-size//2:loc1[0]+size//2]
	tmp2=img[loc2[1]-size//2:loc2[1]+size//2,loc2[0]-size//2:loc2[0]+size//2]
	cv2.imshow('mas',pro_img)
	
	cv2.imwrite(folder+'/template'+str(temp_count)+'.jpg',tmp1)
	temp_count+=1
	cv2.imwrite(folder+'/template'+str(temp_count)+'.jpg',tmp2)
	temp_count+=1

folder='thermal'
images=[folder+'/thermal01.jpg']
template_count=2
for file in images:
	img=cv2.imread(file)
	pro_img=cv2.imread(file)
	extract_template(img,pro_img,TEMPLATE_SIZE,template_count)
	template_count+=2
	cv2.waitKey(0)