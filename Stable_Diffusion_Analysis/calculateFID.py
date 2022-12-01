import os
import cv2
import numpy as np
import shutil

np.random.seed(5824)

#paths = {'real': ('Datasets/extracted_images/faces_COCO/', 'Datasets/extracted_images/faces_COC_tmp'), 
#         'generated': ('images/faces_gen_lowres/', 'images/faces_gen_lowres_tmp/')}
         
paths = {'real': ('Datasets/extracted_images/motion_cocoAPI/LowRes/', 'Datasets/extracted_images/motion_cocoAPI/HighRes_tmp/'), 
         'generated': ('images/motion/', 'images/motion_tmp/')}

# res = {}
#Put number of sample image in each set that we want to do FID between
num_imgs = 1

run = 1

for k in range(run):
	print(k)
	
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
			
			if f == 'generated':
				image = cv2.resize(image,(256,256))
			
			cv2.imwrite(savename, image)
			
			count += 1
			if count >= num_imgs:
				break
	#cmd = "python -m pytorch_fid Datasets/extracted_images/faces_COC_tmp/ images/faces_gen_lowres_tmp/ --device cuda:3"
	cmd = "python -m pytorch_fid Datasets/extracted_images/motion_cocoAPI/HighRes_tmp/ images/motion_tmp/ --device cuda:3"
	
	os.system(cmd)
	
	
	
