import numpy as np
from matplotlib.image import imread
from PIL import Image 
import os
import sys

#def rgb2gray(rgb):
#	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def load_data(file,storage = np.float32, size = (125,125)):
	x = []
	y = []
	if os.path.isfile("pickled/" + file + "_x.npy") and os.path.isfile("pickled/" + file + "_y.npy"):
		x = np.load("pickled/" + file + "_x.npy")
		y = np.load("pickled/" + file + "_y.npy")
		return (x,y)

	with open(file,'r') as train:
		for line in train:
			grp = line.split()
			if len(grp) > 3:
				name1 = grp[0]
				pic1 = grp[1]
				name2 = grp[2]
				pic2 = grp[3]
				img1 = name1 + "_" + str(pic1).zfill(4) + ".jpg"
				img2 = name2 + "_" + str(pic2).zfill(4) + ".jpg"
				im1 = readImage("lfw/" + name1 + "/" + img1,storage,size)
				if im1.size == 0:
					continue
				im2 = readImage("lfw/" + name2 + "/" + img2,storage,size)
				if im2.size == 0: 
					continue
				im = np.concatenate((im1,im2),axis=2)
				x.append(im)
				y.append(0)
			elif len(grp) > 2:
				name1 = grp[0]
				pic1 = grp[1]
				pic2 = grp[2]
				img1 = name1 + "_" + str(pic1).zfill(4) + ".jpg"
				img2 = name1 + "_" + str(pic2).zfill(4) + ".jpg"
				im1 = readImage("lfw/" + name1 + "/" + img1,storage,size)
				if im1.size == 0:
					continue
				im2 = readImage("lfw/" + name1 + "/" + img2,storage,size)
				if im2.size == 0:
					continue
				im = np.concatenate((im1,im2),axis=2)
				x.append(im)
				y.append(1)

	x = normalise(np.array(x))
	y = np.array(y)
	if not os.path.isdir("pickled"):
		os.mkdir("pickled")
	np.save("pickled/" + file + "_x.npy",x)
	np.save("pickled/" + file + "_y.npy",y)

	return (x,y)

def normalise(x):
	return x/255.0

def readImage(image_path,storage,size):
	try:
		img = Image.open(image_path)
	except IOError:
		print("An error occured while reading " + image_path)
		print("Skipping...")
		return []
	img = img.resize(size,Image.ANTIALIAS)
	img = np.array(img).astype(storage)
	return img

