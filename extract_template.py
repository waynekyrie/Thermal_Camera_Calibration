import cv2
import numpy as np
import glob
import string
import time

loc=[]

def click(event, x, y, flags, param):
	# grab references to the global variables
	global loc
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		loc.append((x,y))
	# check to see if the left mouse button was released

def extract_template(tmp_folder,input_img_folder,img_format,tmp_format,tmp_size):
	TEMPLATE_SIZE=tmp_size
	temp_folder=tmp_folder
	img_folder=input_img_folder
	images=glob.glob(img_folder+'*'+img_format)
	temp_count=0
	cv2.namedWindow('image')
	cv2.setMouseCallback('image',click)

	

	for file in images:
		img=cv2.imread(file)
		pro_img=cv2.imread(file)
		cv2.imshow('image',img)
		key=cv2.waitKey(0)&0xFF
		if(key==ord('r')):
			#reset
			img=img.copy()
		elif(key==ord('q')):
			break
	for i in range(0,len(loc)):
		size=TEMPLATE_SIZE
		tmp=np.zeros((size,size),np.uint8)
		location=loc[i]
		tmp=img[location[1]-size//2:location[1]+size//2,location[0]-size//2:location[0]+size//2]
		cv2.imwrite(temp_folder+'/template'+str(temp_count)+tmp_format,tmp)
		temp_count+=1
		loc1=loc[i]
		size=TEMPLATE_SIZE//2
		cv2.rectangle(pro_img,(location[0]-size,location[1]-size),(location[0]+size,location[1]+size),(0,0,255))
	cv2.imshow('Selected_templates',pro_img)
	key=cv2.waitKey(0)&0xFF
	if(key==ord('s')):
		cv2.imwrite(temp_folder+'/temp'+str(temp_count)+'.jpg',pro_img)
	