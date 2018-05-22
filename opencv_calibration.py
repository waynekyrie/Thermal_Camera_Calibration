import cv2
import numpy as np
import glob
import save_calibration as sc

ROW=6
COL=7
images=glob.glob('left*.jpg')
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ROW*COL,3), np.float32)
objp[:,:2] = np.mgrid[0:COL,0:ROW].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    #cv2.imshow("daf",gray)
    #cv2.waitKey(0)
    ret, corners = cv2.findChessboardCorners(gray, (COL,ROW),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #print(corners2)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, (COL,ROW), corners2,ret)
        #cv2.imshow('img',img)
        #v2.waitKey(0)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

sc.save_calibration("builtin_calibration.txt",mtx,dist,rvecs,tvecs)

img=cv2.imread('left03.jpg')
h,w=img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst=cv2.undistort(img,mtx,dist,None,newcameramtx)
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
print( "total error: {}".format(mean_error/len(objpoints)) )
