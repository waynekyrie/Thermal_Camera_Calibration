import cv2
import numpy as np
import glob
import copy

CIRCLE_SIZE=1
def find_template(img,pro_img,tmp_size,template,match_thresh,image_type,board_type):
	img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	existed=[]
	points=[]
	for temp_file in template:
		temp=cv2.imread(temp_file,0)
		w, h = temp.shape[::-1]
		res = cv2.matchTemplate(img_gray,temp,cv2.TM_CCOEFF_NORMED)
		threshold =match_thresh
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
				cor=find_corners(mask,image_type,board_type)
				col=pt[0]+cor[0]
				row=pt[1]+cor[1]
				points.append((row,col))
				existed.append((pt[0],pt[1],cor[0],cor[1]))
	for p in points:
		cv2.circle(pro_img,(int(p[1]),int(p[0])),5,(0,0,255))
	return pro_img,points

def img_maxvalue(img):
	maxi=max(img[0])
	for r in img:
		if(max(r)>maxi):
			maxi=max(r)
	return maxi

def img_minvalue(img):
	mini=min(img[0])
	for r in img:
		if(min(r)<mini):
			mini=min(r)
	return mini

def find_corners(img,image_type,board_type):
	col_cum=0
	col_count=0
	row_cum=0
	row_count=0
	w=img.shape[0]
	h=img.shape[1]

	if(image_type=='thermal'):
		kernel_size=int(w*1/3)
		if(kernel_size%2==0):
			kernel_size+=1
		kernel=np.zeros((kernel_size,kernel_size),dtype=np.float32)
		kernel[:,:]=-1
		kernel[kernel_size//2][kernel_size//2]=kernel_size*kernel_size
		sharp_img = cv2.filter2D(img, -1, kernel)
		min_val=np.amin(sharp_img)
		max_val=np.amax(sharp_img)
		
		if(board_type=='wired'):
			for r in range(h):
				for c in range(w):
					if(sharp_img[r][c]<(min_val+max_val)/2):
						sharp_img[r][c]=0
					else:
						sharp_img[r][c]=1
			kernel[:,:]=0
			kernel[:,kernel_size//2]=1
			kernel[kernel_size//2,:]=1
			kernel[kernel_size//2][kernel_size//2]=0
			out_img = cv2.filter2D(sharp_img, -1, kernel)
			max_val=np.amax(out_img)
			for r in range(len(img)):
				for c in range(len(img)):
					if(r>=int(0.75*len(img)) or r<=int(0.25*len(img)) or c>=int(0.75*len(img)) or c<=int(0.25*len(img))):
						out_img[r][c]=0
					else:
						if(out_img[r][c]>=max_val):
							col_count+=1
							row_count+=1
							row_cum+=r
							col_cum+=c
							out_img[r][c]=255
						else:
							out_img[r][c]=0
		elif(board_type=='chessboard'):
			for r in range(h):
				for c in range(w):
					if(sharp_img[r][c]<(min_val+max_val)/2):
						sharp_img[r][c]=0
					else:
						sharp_img[r][c]=255
			kernel[:,:]=0
			kernel[:,kernel_size//2]=1
			kernel[kernel_size//2,:]=1
			kernel[kernel_size//2][kernel_size//2]=0
			out_img = cv2.filter2D(sharp_img, -1, kernel)
			max_val=np.amax(out_img)
			for r in range(len(img)):
				for c in range(len(img)):
					if(r>=int(0.75*len(img)) or r<=int(0.25*len(img)) or c>=int(0.75*len(img)) or c<=int(0.25*len(img))):
						out_img[r][c]=0
					else:
						if(out_img[r][c]>=max_val):
							col_count+=1
							row_count+=1
							row_cum+=r
							col_cum+=c
							out_img[r][c]=255
						else:
							out_img[r][c]=0
		if(col_count==0 or row_count==0):
			cor=(img.shape[0]/2,img.shape[1]/2)
		else:
			cor=(col_cum/col_count,row_cum/row_count)	
	elif(image_type=='visual'):
		corners=cv2.goodFeaturesToTrack(img,maxCorners=100,qualityLevel=0.05,minDistance=0,
			useHarrisDetector=0)
		for c in corners:
			tmp=c[0]
			if(tmp[1]>=0.25*img.shape[0] and tmp[1]<=0.75*img.shape[0] and tmp[0]>=0.25*img.shape[0] and tmp[0]<=0.75*img.shape[0]):
				col_count+=1
				row_count+=1
				row_cum+=tmp[1]
				col_cum+=tmp[0]
		cor=(col_cum/col_count,row_cum/row_count)
	else:
		print('Unsupport img type')
		cor=(img.shape[0]/2,img.shape[1]/2)
	
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

def top_p(points,board_r,board_c):
	maxr=1000;
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
	for i in range(0,min(board_r,board_c)):
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
			if(p[1]<maxc):
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

def reorder_points(file,points,board_r,board_c,dist_thresh):
	p_start=top_p(points,board_r,board_c)
	img=cv2.imread(file)
	min_dist1=1000
	min_dist2=1000
	
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
	line1,s1=points_on_line(points,p_start,p1,dist_thresh)
	line2,s2=points_on_line(points,p_start,p2,dist_thresh)
	
	if((len(line1)!=board_r or len(line2)!=board_c) and (len(line2)!=board_r or len(line1)!=board_c)):
		return [line1]
	if(len(line2)==len(line1)):
		if(s1==None or (s2!=None and abs(s2)<abs(s1))):
			s=s2
			row_axis=copy.copy(line1)
			col_axis=copy.copy(line2)
		elif(s2==None or (s1!=None and abs(s1)<abs(s2))):
			s=s1
			row_axis=copy.copy(line2)
			col_axis=copy.copy(line1)
		else:
			row_axis=copy.copy(line1)
			col_axis=copy.copy(line2)
			s=s2
	else:
		if(len(line1)>len(line2)):
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
			if(p not in row_axis):
				if(s==None):
					if(abs(p[1]-r[1])<dist_thresh):
						dist=p_to_p(r,p)
						if(dist<min_dist_to_r and p!=r):
							p_next=p
							min_dist_to_r=dist
				else:
					a=s
					b=-1
					c=(-1)*s*r[1]+r[0]
					d=abs(a*p[1]+b*p[0]+c)/np.sqrt(np.square(a)+np.square(b))
					if(d<dist_thresh):
						dist=p_to_p(r,p)
						if(dist<min_dist_to_r and p!=r):
							min_dist_to_r=dist
							p_next=p
		tmp_line,tmp_s=points_on_line(points,r,p_next,dist_thresh)
			
		p_far=farthest(tmp_line,r)
		tmp_line1,tmp_s1=points_on_line(points,r,p_far,dist_thresh)
		for p in tmp_line1:
			if(p not in tmp_line):
				tmp_line.append(p)
		dist=[]
		for p in tmp_line:
			if(p==r):
				dist.append(0)
			else:
				d=p_to_p(r,p)
				dist.append(d)
		tmp_line=sort_points_on_line(tmp_line,dist)
		line.append(tmp_line)	
	return line

def resize_matrix(matrix,board_r,board_c):
	pt=[]
	c_max=0
	c_min=1000
	for r in matrix:
		if(len(r)>c_max):
			c_max=len(r)
		if(len(r)<c_min):
			c_min=len(r)
	#required size larger than detected board size
	if(board_r>=len(matrix) and board_c>=c_max):
		for r in matrix:
			for c in r:
				pt.append([c])
		return pt
	r_start=0
	r_end=len(matrix)
	c_start=0
	c_end=c_min
	if(board_r<=len(matrix)):
		r_diff=len(matrix)-board_r
		r_start=(r_diff-1)//2+1
		r_end=len(matrix)-r_diff//2
	if(board_c<=c_min):
		c_diff=c_min-board_c
		c_start=(c_diff-1)//2+1
		c_end=c_min-c_diff//2
	for r in range(r_start,r_end):
		for c in range(c_start,c_end):
			pt.append([matrix[r][c]])
	return pt


def draw_board(img,pt,board_r,board_c,arrow=False):
	count=0
	first=1
	BOARD_SIZE_C=board_c
	BOARD_SIZE_R=board_r
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
		if(arrow):
			if(first):
				first=0
				pre=p
			cv2.arrowedLine(img,(int(pre[1]),int(pre[0])),(int(p[1]),int(p[0])),color)
		cv2.circle(img,(int(p[1]),int(p[0])),CIRCLE_SIZE,color)
		pre=p
		count+=1

def get_square_img_size(img):
	return img.shape[0]

def find_crosspoints(file,template,thresh,board_r,board_c,image_type,board_type):

	img=cv2.imread(file)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	pro_img=cv2.imread(file)
	template_size=get_square_img_size(cv2.imread(template[0]))
	pro_img,points=find_template(img,pro_img,template_size,template,thresh,image_type,board_type)
	
	#check amount of found crosspoints
	if(len(points)!=len(set(points)) or len(set(points))!=board_r*board_c):
		return False,[(0,0)]

	for dist in range(0,30):
		matrix=reorder_points(file,points,board_r,board_c,dist)
		flag=True
		#check every row has same amount of points
		count=0
		for l in matrix:
			count+=len(l)
			if(len(l)!=board_c):
				flag=False
		if(flag==True and len(matrix)==board_r and count==board_r*board_c):
			break
	if(len(matrix)!=board_r or flag==False):
		return False,[(0,0)]#Detected no feature point
	pt=resize_matrix(matrix,board_r,board_c)
	#check if detected points satisfy requirement
	if(len(pt)==board_r*board_c):
		ret=True
	else:
		ret=False
	return ret,pt


def opencv_find_crosspts(image_files,objp,row,col,draw_b=False):
    imgpoints=[]
    obj=[]
    count=0
    missed=[]
    valid=[]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for fname in image_files:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (col,row),None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            points=[]
            for r in corners2:
            	tmp=r[0]
            	points.append((tmp[1],tmp[0]))
            for dist in range(0,30):
            	matrix=reorder_points(fname,points,row,col,dist)
            	flag=True
            	pre=col
            	#check every row has same amount of points
            	for l in matrix:
            		if(len(l)!=pre):
            			flag=False
            	if(flag==True and len(matrix)==row):
            		break
            pt=resize_matrix(matrix,row,col)
            if(draw_b):
            	draw_board(img,pt,row,col,arrow=True)
            	cv2.imshow('draw_board',img)
            if(cv2.waitKey(0)&0xFF==ord('q')):
            	break
            for i in range(0,len(pt)):
            	for j in range(0,len(pt[i])):
            		pt[i][j]=(np.float32(pt[i][j][1]),np.float32(pt[i][j][0]))
            cor=np.array(pt)
            imgpoints.append(cor)
            obj.append(objp)
            count+=1

    imgpoints=np.array(imgpoints)
    obj=np.array(obj)
    return imgpoints,obj
