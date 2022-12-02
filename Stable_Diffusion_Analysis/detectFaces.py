import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
#import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
#from PIL import ImageFont
import cv2
import sys
import json

import mtcnn
from mtcnn.mtcnn import MTCNN
from numpy import asarray
import matplotlib.pyplot as plt
import time

# Simple python package to shut up Tensorflow warnings and logs.
import silence_tensorflow.auto


def filterImage(filename, savefilename, required_size=(250, 250)):
	image = Image.open(filename)
	# convert to RGB, if needed

	image = image.convert('RGB')
	# convert to array
	pixels = np.asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image

	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	#print(results)

	if len(results) >=1:
		x1, y1, width, height = results[0]['box']
		print((x1,y1,width,height))

		if height-width>=15:
		# bug fix
			x1, y1 = abs(x1), abs(y1)
			x2, y2 = x1 + width, y1 + height
			# extract the face
			face = pixels[y1:y2, x1:x2]
			# resize pixels to the model size
			image = Image.fromarray(face)
			image = image.resize(required_size)
			face_array = np.asarray(image)
			plt.imsave(savefilename,face_array)
			return 1
	return 0

dataDir = 'images/'

src = dataDir + "faces/"
dest = dataDir + "faces_gen_lowres/"


totalimages = 10000
filtered=0
for i in range(totalimages):

	start = time.time()
	filename = src + "image" + str(i) + '.jpg'
	savefilename = dest + "image_" + str(i) + '.jpg'
	
	if filterImage(filename,savefilename) == 1:
		filtered += 1
	else:
		print(f"Failedto filter {filename}")
	#print(f"Time taken = {time.time() - start} sec." )
	#break

print(f"Total filtered image count = {filtered}")
	

