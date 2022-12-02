import os
import cv2
import numpy as np
import shutil
import mtcnn
from PIL import Image
from mtcnn.mtcnn import MTCNN

np.random.seed(5824)

#paths = {'real': ('Datasets/extracted_images/faces_COCO/', 'Datasets/extracted_images/faces_COC_tmp'), 
#         'generated': ('images/faces_gen_lowres/', 'images/faces_gen_lowres_tmp/')}
         
paths = {'real': ('Datasets/extracted_images/motion_cocoAPI/LowRes/', 'Datasets/extracted_images/motion_cocoAPI/HighRes_tmp/'), 
         'generated': ('images/motion/', 'images/motion_tmp/')}

# res = {}
num_imgs = 5

for k in range(1):
	print(k)
	turn = 1
	for f in ['real', 'generated']: #range(2):
		img_path, save_path = paths[f]
		files = os.listdir(img_path)
		
		#np.random.shuffle(files)
		
		if os.path.exists(save_path):
			shutil.rmtree(save_path)    
		os.mkdir(save_path)
		
		#print(save_path)
		
		count = 0
		
		
		for f_name in files:
			imagename = os.path.join(img_path,f_name)
			savename = os.path.join(save_path, f_name)
			print(imagename)
			print(savename)
			
			image = cv2.imread(imagename)
			imagePI = Image.open(imagename)
			
			if f == 'generated':
				#image = cv2.resize(image,(256,256))
				#cv2.imwrite(savename, image)
				#image = cv2.imread(savename)
				#imagePI = Image.open(savename)
				pass
			else:
				imagePI = Image.open(imagename)
				
			imagePI = imagePI.convert('RGB')
			
			detector = MTCNN()
			pixels = np.asarray(imagePI)
			
			results = detector.detect_faces(pixels)
			
			x1, y1, width, height = results[0]['box']
			print((x1,y1,width,height))
			x1 -= int(80*turn)
			y1 -= int(80*turn)
			
			if x1<0:
				x1 = 0
			if y1<0:
				y1=0
			height = int(120*turn)
			width = int(120*turn)
			x2, y2 = x1 + width, y1 + height
			
			if x2 > 256*turn:
				x1 = 256*turn
			if y1 > 256*turn:
				y1 = 256*turn
			
			print((x1,y1,x2,y2))
			
			motion = image[y1:y2, x1:x2]
			
			cv2.imwrite(savename, motion)
			
			count += 1
			
			if count >= num_imgs:
				break
		turn += 1
	#cmd = "python -m pytorch_fid Datasets/extracted_images/faces_COC_tmp/ images/faces_gen_lowres_tmp/ --device cuda:3"
	cmd = "python -m pytorch_fid Datasets/extracted_images/motion_cocoAPI/HighRes_tmp/ images/motion_tmp/ --device cuda:3"
	
	os.system(cmd)
	
	
	
