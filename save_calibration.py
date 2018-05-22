
def save_calibration(filename,mtx,dist,rvecs,tvecs):
	file=open(filename,"w")
	file.write("Camera Matrix\n")
	file.write(str(mtx))
	file.write("\n\nDistortion Factors\n")
	file.write(str(dist))
	file.write("\n\nRot Vector\n")
	file.write(str(rvecs))
	file.write("\n\nTran Vector\n")
	file.write(str(tvecs))
	file.close()