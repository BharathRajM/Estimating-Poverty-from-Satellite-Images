# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:01:15 2020

@author: 91782
"""
import os
import shutil
import pandas as pd
import numpy as np

classify_path='E:/Datasets/Segregate_Folder_2/'  
#Create this folder and update path. 
#Train and Test folders and class folders will be created inside it

data_path = 'E:/Datasets/Google_Imgs/'
#Paste all images in dataset here. Images will be COPIED from here to classify_path

gps_ntl_path = 'E:\\Datasets\\ntl_wealth_1.csv'
#Place the csv file containing gps data. It must be a table of 3 columns.
#Column Index are 'Latitude', 'Longitude', 'ntl'. Match case exactly in csv

def segregate_files_to_folders(file_path):
	train_folder = os.path.join(classify_path,'Train')
	test_folder = os.path.join(classify_path,'Test')
	if not os.path.exists(os.path.join(classify_path,'Train')):
		#Create Train folder if not present
		os.mkdir(train_folder)
	if not os.path.exists(os.path.join(classify_path,'Test')):
		#Create Test folder if not present
		os.mkdir(test_folder)
	if os.path.exists(file_path):
		file = open(file_path)
		#Read gps coord vs ntl intensity data into a dataframe
		gi_df = pd.read_csv(file_path)
		for index in range(0,gi_df.shape[0]):
			try:
				ntl_value = int(round(gi_df['ntl'][index],0))
				#If Night light intensity is less than 8, it is classified as low intensity
				#If Night light intensity is between 8 and 35, it is classified as medium intensity
				#If Night light intensity is greater than 35 , it is classified as high intensity
				if ntl_value <= 8:
					ntl_value = 'Low Intensity'
				elif (ntl_value > 8 and ntl_value <=35):
					ntl_value = 'Medium Intensity'
				else:
					ntl_value = 'High Intensity'
				#Train test split is set at 0.8
				if index < (gi_df.shape[0] * 0.8):
					ntl_folder = os.path.join(train_folder,str(ntl_value))
				else:
					ntl_folder = os.path.join(test_folder,str(ntl_value))
				#If intensity class folder is not present, create folder
				if not os.path.exists(ntl_folder):
					os.mkdir(ntl_folder)
				#Create file name from csv and copy from src to classified folder
				file_name = str(gi_df['Latitude'][index])+'_'+str(gi_df['Longitude'][index])+'.png'
				if not os.path.exists(os.path.join(ntl_folder,file_name)):
					shutil.copy(os.path.join(data_path,file_name),ntl_folder)
			#Exception handling done to continue segregating even if few files fail
			except Exception as ex:
				#Write error message to log
				with open('copy_log.txt','w+') as f:
					f.write('Failed to copy '+file_name+'. Exception: '+str(ex))
		
def main():
	segregate_files_to_folders(gps_ntl_path)
	
if __name__ == '__main__':
	main()
