import cv2
import numpy as np
import glob
import save_calibration as sc
import cam_calibration as cc
import time
import copy

img_folder='thermal/thermal_0525_1/'
template_folder='thermal/0.0606template/'
BOARD_SIZE_R=7
BOARD_SIZE_C=7
MATCH_THRESH=0.8
TEMPLATE_SIZE=26

#load calibration images and templates
images=glob.glob(img_folder+'20*.jpg')
template=glob.glob(template_folder+'template*.jpg')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((BOARD_SIZE_R*BOARD_SIZE_C,3), np.float32)
objp[:,:2] = np.mgrid[0:BOARD_SIZE_C,0:BOARD_SIZE_R].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

count=0
img_count=0
for fname in images:
	t1=time.time()
	img=cv2.imread(fname)  
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,corners = cc.find_crosspoints(fname,template,MATCH_THRESH,BOARD_SIZE_R,BOARD_SIZE_C)
	#cc.draw_board(img,corners,BOARD_SIZE_R,BOARD_SIZE_C)
	t2=time.time()
	t_cost=t2-t1
    # If found, add object points, image points (after refining them)
	if ret == True:
		for i in range(0,len(corners)):
			for j in range(0,len(corners[i])):
				corners[i][j]=(np.float32(corners[i][j][1]),np.float32(corners[i][j][0]))
		cor=np.array(corners)
		objpoints.append(objp)
		imgpoints.append(cor)
		img_count+=1
	else:
		count+=1
		print(fname)
print("Num of invalid img: {}".format(count))
print("Num of used img: {}".format(img_count))
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#########################################
#save calibration results
#########################################
sc.save_calibration("calibration.txt",mtx,dist,rvecs,tvecs)


#########################################
#display undistorted image
#########################################

for fname in images:
	img=cv2.imread(fname)
	h,w=img.shape[:2] 
	newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
	dst=cv2.undistort(img,mtx,dist,None,newcameramtx)
	x, y, w, h = roi
	#dst = dst[y:y+h, x:x+w]
	cv2.imshow('dst',dst)
	cv2.waitKey(0)


#########################################
#re-projection error
#########################################
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
 
print( "mean error: {}".format(mean_error/len(objpoints)))

cv2.destroyAllWindows()