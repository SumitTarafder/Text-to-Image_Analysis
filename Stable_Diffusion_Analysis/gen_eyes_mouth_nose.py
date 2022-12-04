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


def filterMouth(filename, savefilename, required_size=(300, 150)):
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
		mouth_width = abs(results[0]['keypoints']['mouth_left'][0] - results[0]['keypoints']['mouth_right'][0])
	else:
		return 0

	try:
		if mouth_width>=100:
			#print(results[0])
			xmouth= results[0]['keypoints']['mouth_left'][0]-5
			ymouth = results[0]['keypoints']['mouth_left'][1]-10
			width = results[0]['keypoints']['mouth_right'][0] +5
			height = int(results[0]['keypoints']['mouth_left'][1] + results[0]['keypoints']['mouth_left'][0]/2)-5
			
			mouth = pixels[ymouth:height, xmouth:width]
			# resize pixels to the model size
			
			image = Image.fromarray(mouth)
			image = image.resize(required_size)
			mouth_array = np.asarray(image)
			#plt.imshow(mouth_array)
			plt.imsave(savefilename,mouth_array)
			return 1
	except:
		None
	return 0

def filterEye(filename, savefilename, required_size = (300,100),left_eye_corner = 20, right_eye_corner = 230):
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
		left_right_eye_x_diff = results[0]['keypoints']['right_eye'][0] - results[0]['keypoints']['left_eye'][0]
		left_right_eye_y_diff = abs(results[0]['keypoints']['right_eye'][1] - results[0]['keypoints']['left_eye'][1])
	else:
		return 0

	try:
		if left_right_eye_x_diff>=100 and left_right_eye_y_diff < 8:
			#print(results[0])
			xeye = left_eye_corner
			yeye = int((results[0]['keypoints']['left_eye'][1]+results[0]['keypoints']['right_eye'][1])/2 -10)
			width = right_eye_corner
			height = int((yeye+results[0]['keypoints']['nose'][1])/2)
			eyes = pixels[yeye:height, xeye:width]
			
			image = Image.fromarray(eyes)
			image = image.resize(required_size)
			eyes_array = np.asarray(image)
			plt.imsave(savefilename,eyes_array)
			
			return 1
	except:
		None
	return 0

def filterNose(filename, savefilename, required_size = (200,200)):
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
	
	try:
		if len(results)>=1:
			#print(results[0])
			
			xnose= results[0]['keypoints']['nose'][0]-35
			ynose = results[0]['keypoints']['nose'][1]+15
			
			width = results[0]['keypoints']['nose'][0] +35
			
			height = int((results[0]['keypoints']['left_eye'][1]+results[0]['keypoints']['right_eye'][1])/2)+10
			nose = pixels[height:ynose, xnose:width]
			# resize pixels to the model size
			image = Image.fromarray(nose)
			image = image.resize(required_size)
			nose_array = np.asarray(image)
			plt.imshow(nose_array)
			
			plt.imsave(savefilename,nose_array)
			return 1
	except:
		None
	return 0

dataDir = 'images/'
dataDir2 = "Text-to-Image_Analysis/Stable_Diffusion_Analysis/"

#src="Datasets/extracted_images/faces_COCO/"
src = dataDir + "faces_gen_lowres/"

dest1 = dataDir + "gen_mouth/"
dest2 = dataDir + "gen_eyes/"
dest3 = dataDir2 + "extracted_noses_mine/"
dest4 = dataDir2 + "Gen_nose_Diffusion/"

files = os.listdir(src)
eyesgenerated = 0
mouthgenerated = 0
nosegenerated = 0

print(len(files))

for f_name in files:

	start = time.time()
	filename = src + f_name
	savefilenamemouth = dest1 + "image_" + str(mouthgenerated) + '.jpg'
	savefilenameeye = dest2 + "image_" + str(eyesgenerated) + '.jpg'
	#savefilenamnose = dest3 + "image_" + str(nosegenerated) + '.jpg'
	savefilenamnose = dest4 + "image_" + str(nosegenerated) + '.jpg'
	
	#print(f"starting {filename}")
	#print(savefilename)
	
	'''
	if filterMouth(filename,savefilenamemouth) == 1:
		mouthgenerated += 1
	else:
		print(f"Failedto filter mouth {filename}")
	
	#print(f"Time taken = {time.time() - start} sec." )
	if filterEye(filename,savefilenameeye) == 1:
		eyesgenerated += 1
	else:
		print(f"Failedto filter eye {filename}")
	'''
	
	#if eyesgenerated == 1 and mouthgenerated == 1:
		#break
	if filterNose(filename,savefilenamnose) == 1:
		nosegenerated += 1
	else:
		print(f"Failedto filter nose {filename}")
	
	
	print(f"finished {filename}")

print(f"Total filtered nose image count = {nosegenerated}")
print(f"Total filtered mouth image count = {mouthgenerated}")
print(f"Total filtered eye image count = {eyesgenerated}")

