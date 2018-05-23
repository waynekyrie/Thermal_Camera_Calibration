import cv2
import numpy as np
import glob
import save_calibration as sc
import time
import copy

BOARD_SIZE_R=7
BOARD_SIZE_C=7
CIRCLE_SIZE=5
MATCH_THRESH=0.85
TEMPLATE_SIZE=26
folder='thermal/'

def find_template(img,pro_img,tmp_size,template):
	img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	existed=[]
	points=[]
	for temp_file in template:
		temp=cv2.imread(temp_file,0)
		w, h = temp.shape[::-1]
		res = cv2.matchTemplate(img_gray,temp,cv2.TM_CCOEFF_NORMED)
		threshold =MATCH_THRESH
		loc = np.where( res >= threshold)
		mask=np.zeros((tmp_size,tmp_size),np.float32)
		#pt top-left corner of the matching window
		for pt in zip(*loc[::-1]):
			new=True
			for pre in existed:
				if(np.sqrt(np.square(pt[0]-pre[0])+np.square(pt[1]-pre[1]))<15):
					if(res[pt[1]][pt[0]]>res[pre[1]][pre[0]]):
						points.remove((pre[1]+pre[3],pre[0]+pre[2]))
						existed.remove(pre)
					else:	
						new=False
					break
			if(new):
				mask=copy.copy(img_gray[pt[1]:pt[1]+tmp_size,pt[0]:pt[0]+tmp_size])
				cor=find_corners(mask)
				col=pt[0]+cor[0]
				row=pt[1]+cor[1]
				cv2.circle(pro_img,(col,row),CIRCLE_SIZE,(0,0,255))
				points.append((row,col))
				existed.append((pt[0],pt[1],cor[0],cor[1]))
	return pro_img,points

def find_corners(img):
	'''
	dst=cv2.cornerHarris(img,2,3,0.04)
	ret,dst=cv2.threshold(dst,0.01*dst.max(),255,0)
	dst=np.uint8(dst)
	cor=[]
	img[dst>0.01*dst.max()]=0
	c_count=0
	r_count=0
	w=img.shape[1]
	h=img.shape[0]
	for r in range(0,len(img)):
		for c in range(0,len(img[r])):
			if(img[r][c]==0 and (r>w*(1/4) and r<w*(3/4)) and (c>h*(1/4) and c<h*(3/4))):
				cor.append((c,r))
				c_count+=c
				r_count+=r
	cor=(c_count//(len(cor)),r_count//(len(cor)))
	'''
	cor=(13,13)
	return cor

def farthest(points,pt):
	max_d=0
	max_p=pt
	for p in points:
		dist=p_to_p(p,pt)
		if(dist>max_d):
			max_d=dist
			max_p=p
	return max_p

def top_p(points):
	maxr=480;
	maxc=0;
	tlp=(0,0)
	pt=[]
	for p in points:
		#top
		if(p[0]<maxr):
			maxr=p[0]
			maxc=p[1]
			tlp=p
	maxr=tlp[0]
	minr=tlp[0]
	pt.append(tlp)
	for i in range(0,min(BOARD_SIZE_R,BOARD_SIZE_C)):
		for p in points:
			if(p!=tlp):
				if(p[0]<maxr+5 and p[0]>minr-5):
					pt.append(p)
					maxr=max(p[0],maxr)
					minr=min(p[0],minr)
					break
	if(len(pt)>1):
		p1=farthest(pt,pt[0])
		pt,s=points_on_line(points,pt[0],p1,15)
		for p in pt:
			if(p[1]>maxc):
				maxc=p[1]
				tlp=p
	else:
		tlp=pt[0]
	return tlp

def p_to_p(p1,p2):
	dist=np.sqrt(np.square(p1[0]-p2[0])+np.square(p1[1]-p2[1]))
	return dist

def points_on_line(points,p1,p2,thresh):
	if(p1[1]!=p2[1]):
		s=(p1[0]-p2[0])/(p1[1]-p2[1])
		c=(-s)*p2[1]+p2[0]
		a=s
		b=-1
	else:
		a=1
		b=0
		c=(-1)*p2[1]
		s=None
	l=[]
	dist=[]
	line=[]
	for p in points:
		d=abs(a*p[1]+b*p[0]+c)/np.sqrt(np.square(a)+np.square(b))
		if(d<thresh):
			line.append(p)
			dist_points=p_to_p(p,p1)
			dist.append(dist_points)
	line=sort_points_on_line(line,dist)
	return line,s

def next_min_index(dist,cur_min,existed_index):
	mini=max(dist)+1
	min_index=0
	for i in range(0,len(dist)):
		if(dist[i]<mini and dist[i]>=cur_min):
			if(i not in existed_index):
				min_index=i
				mini=dist[i]

	return min_index,mini

def sort_points_on_line(line,dist):
	cur_min=-1
	l=[]
	
	existed_index=[]
	for i in range(0,len(line)):
		index,tmp=next_min_index(dist,cur_min,existed_index)
		existed_index.append(index)
		l.append(line[index])
		cur_min=tmp
	return l

def reorder_points(file,points):
	p_start=top_p(points)
	img=cv2.imread(file)
	min_dist1=200
	min_dist2=200
	p1=(0,0)
	p2=(0,0)
	for p in points:
		if(p!=p_start):
			dist=p_to_p(p,p_start)
			if(dist<min_dist1):
				min_dist2=min_dist1
				min_dist1=dist
				p2=p1
				p1=p
			elif(dist<min_dist2):
				min_dist2=dist
				p2=p
	
	line=[]
	line1,s1=points_on_line(points,p_start,p1,15)
	line2,s2=points_on_line(points,p_start,p2,15)
	
	if(len(line1)>=len(line2)):
		s=s1
		row_axis=copy.copy(line2)
		col_axis=copy.copy(line1)
	else:
		s=s2
		col_axis=copy.copy(line2)
		row_axis=copy.copy(line1)
	p_next=(0,0)
	l=[]
	for r in row_axis:
		min_dist_to_r=1000
		
		for p in points:
			if(s==None):
				if(abs(p[1]-r[1])<20):
					dist=p_to_p(r,p)
					if(dist<min_dist_to_r and p!=r):
						p_next=p
						min_dist_to_r=dist
			else:
				a=s
				b=-1
				c=(-1)*s*r[1]+r[0]
				d=abs(a*p[1]+b*p[0]+c)/np.sqrt(np.square(a)+np.square(b))
				if(d<15):
					dist=p_to_p(r,p)
					if(dist<min_dist_to_r and p!=r):
						min_dist_to_r=dist
						p_next=p
		tmp_line,tmp_s=points_on_line(points,r,p_next,15)
		p_far=farthest(tmp_line,r)
		tmp_line1,tmp_s1=points_on_line(points,r,p_far,15)
		for p in tmp_line1:
			if(p not in tmp_line):
				tmp_line.append(p)
		line.append(tmp_line)
	return line

def resize_matrix(matrix):
	pt=[]
	c_max=0
	c_min=1000
	for r in matrix:
		if(len(r)>c_max):
			c_max=len(r)
		if(len(r)<c_min):
			c_min=len(r)
	if(BOARD_SIZE_R>=len(matrix) and BOARD_SIZE_C>=c_max):
		for r in matrix:
			for c in r:
				pt.append([c])
		return pt
	r_start=0
	r_end=len(matrix)
	c_start=0
	c_end=c_min

	if(BOARD_SIZE_R<len(matrix)):
		r_diff=len(matrix)-BOARD_SIZE_R
		r_start=(r_diff-1)//2+1
		r_end=len(matrix)-r_diff//2
	if(BOARD_SIZE_C<c_max):
		c_diff=c_min-BOARD_SIZE_C
		c_start=(c_diff-1)//2+1
		c_end=c_min-c_diff//2

	for r in range(r_start,r_end):
		for c in range(c_start,c_end):
			pt.append([matrix[r][c]])
	return pt


def draw_board(img,pt):
	count=0
	for p in pt:
		if(count//BOARD_SIZE_C==3):
			count=0
		if(count//BOARD_SIZE_C==0):
			color=(0,0,255)
		elif(count//BOARD_SIZE_C==1):
			color=(0,255,0)
		elif(count//BOARD_SIZE_C==2):
			color=(255,0,0)
		p=p[0]
		
		cv2.circle(img,(p[1],p[0]),5,color)
		count+=1

def find_crosspoints(file,template):
	img=cv2.imread(file)
	pro_img=cv2.imread(file)
	p_img=cv2.imread(file)
	template_size=TEMPLATE_SIZE


	pro_img,points=find_template(img,pro_img,template_size,template)
	#cv2.imshow('asf', pro_img)
	#cv2.waitKey(0)
	matrix=reorder_points(file,points)
	pt=resize_matrix(matrix)
	draw_board(p_img,pt)
	if(len(pt)>0):
		ret=True
	else:
		ret=False
	return ret,p_img,pt

images=glob.glob('left*.jpg')
images=glob.glob(folder+'thermal*.jpg')
template=glob.glob('template/template*.jpg')
template=glob.glob(folder+'template*.jpg')



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((BOARD_SIZE_R*BOARD_SIZE_C,3), np.float32)
objp[:,:2] = np.mgrid[0:BOARD_SIZE_C,0:BOARD_SIZE_R].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for fname in images:
	img=cv2.imread(fname)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
	ret, pro_img,corners = find_crosspoints(fname,template)
	cv2.imshow('daf',pro_img)
	cv2.waitKey(0)
    # If found, add object points, image points (after refining them)
	if ret == True:
		for i in range(0,len(corners)):
			for j in range(0,len(corners[i])):
				corners[i][j]=(np.float32(corners[i][j][0]),np.float32(corners[i][j][1]))
		cor=np.array(corners)
		objpoints.append(objp)
		imgpoints.append(cor)
		#cv2.imshow('p',pro_img)
		#cv2.waitKey(0)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)



img=cv2.imread(folder+'thermal01.jpg')
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
cv2.destroyAllWindows()
