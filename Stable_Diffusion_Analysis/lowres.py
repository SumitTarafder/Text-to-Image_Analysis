import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
#import matplotlib.pyplot as plt
#from PIL import Image, ImageDraw
#from PIL import ImageFont
import cv2
import sys
import json


def resizeImage(filename, savefilename, low_res = 256):
	image = cv2.imread(filename)
	image_res = cv2.resize(image,(low_res,low_res))
	
	cv2.imwrite(savefilename, image_res)
	#print(savefilename)


dataDir = 'Datasets/'
highres = "HighRes"
lowres = "LowRes"

savedirhigh = dataDir + "extracted_images/motion_cocoAPI/" + highres + "/"
savedirlow = dataDir + "extracted_images/motion_cocoAPI/" + lowres + "/"


totalimages = 10000

for i in range(totalimages):

	filename = savedirhigh + "image_" + str(i) + '.jpg'
	savefilename = savedirlow + "image_" + str(i) + '.jpg'
	
	resizeImage(filename,savefilename)
	#break
	







