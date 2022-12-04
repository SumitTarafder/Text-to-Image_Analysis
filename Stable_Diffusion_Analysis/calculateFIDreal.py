import os
import cv2
import numpy as np
import shutil
np.random.seed(5824)

for k in range(10):
	# split real faces into two folders 
	direc="Text-to-Image_Analysis/Stable_Diffusion_Analysis/extracted_noses"
	files = os.listdir(direc)
	
	np.random.shuffle(files)
	
	if os.path.exists('Datasets/extracted_images/real_tmp/tmp1/'):
		shutil.rmtree('Datasets/extracted_images/real_tmp/tmp1/')  
	os.mkdir('Datasets/extracted_images/real_tmp/tmp1/')    
	
	if os.path.exists('Datasets/extracted_images/real_tmp/tmp2/'):
		shutil.rmtree('Datasets/extracted_images/real_tmp/tmp2/')  
	os.mkdir('Datasets/extracted_images/real_tmp/tmp2/')    
	
	for i, f in enumerate(files[:5000]):
		if i%2:
			shutil.copy(f'{direc}/{f}',f'Datasets/extracted_images/real_tmp/tmp1/{f}')
		else:
			shutil.copy(f'{direc}/{f}',f'Datasets/extracted_images/real_tmp/tmp2/{f}')    
			
	cmd = "python -m pytorch_fid Datasets/extracted_images/real_tmp/tmp1/ Datasets/extracted_images/real_tmp/tmp2/ --device cuda:3"
	
	os.system(cmd)
