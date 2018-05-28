import numpy as np
import cv2
import glob

board_r=11
board_c=4

def border_points(points):
	left=1000
	l_pt=[0,0]
	bot=0
	b_pt=[0,0]
	up=1000
	u_pt=[0,0]
	right=0
	r_pt=[0,0]
	for pt in points:
		x=pt[0]
		y=pt[1]
		if(x<left):
			left=x
			l_pt=pt
		if(x>right):
			right=x
			r_pt=pt
		if(y<up):
			up=y
			u_pt=pt
		if(y>bot):
			bot=y
			b_pt=pt
	return (u_pt,b_pt,l_pt,r_pt)


def extract_circles(img,pro_img,radius_range):
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	circles=[[]]
	p1=60
	while(len(circles[0])!=board_r*board_c and p1>=0):
		circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,15,
                            param1=p1,param2=20,minRadius=radius_range[0],maxRadius=radius_range[1])
		p1-=1
	circles=circles[0]
	for cir in circles:
		cv2.circle(pro_img,(cir[0],cir[1]),1,(0,0,255))
	return pro_img,circles



file=glob.glob('thermal_circle/20*.jpg')
for fname in file:
	img=cv2.imread(fname)
	pro_img=cv2.imread(fname)
	print(fname)
	pro_img,circles=extract_circles(img,pro_img,(5,40))
	border_pt=border_points(circles)
	for p in border_pt:
		cv2.circle(pro_img,(p[0],p[1]),5,(0,255,0))
	cv2.imshow('adfaf',pro_img)
	cv2.waitKey(0)