# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:24:35 2020

@author: Adithya

Code for all API calls made from python.
"""

import requests
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
# Path for directory with all datasets
dataset_dir = 'E:\\Datasets\\'
# Path for directory with all Google Satellite Images
day_img_dir = 'E:\\Datasets\\Google_Imgs\\'
# Path for log file for API requests
log_file = 'D:\\Deep Learning\\Project2\\Code\\api_req_log.txt'

def logger(log):
	#writes API request logs to file
	f = open(log_file,'w+')
	f.write(log)
	f.close()
	
def get_img(latitude,longitude,api_key):
	
	latitude =float(latitude)
	longitude = float(longitude)
	#create file name from LAtitude and Longitude
	file_name = str(latitude)+'_'+str(longitude)+'.png'
	#create file path from file name
	file_path = os.path.join(day_img_dir,file_name)
	#Google Static Maps API url
	url = "https://maps.googleapis.com/maps/api/staticmap?v=3.35&"
	#Zoom level set for 10 km cluster area
	zoom = 17
	if not os.path.exists(file_path):
		try:
			#Define URL with latitude , longitude ,zoom level and no styling
			#We take 400x450 as the image size and crop out last few rows to remove Google Watermark
			
			url = url + '&center='+str(latitude)+','+str(longitude)+'&zoom='+str(zoom)+'&size=400x450&key='+api_key+'&sensor=false&maptype=satellite&style=feature:poi|element:labels|visibility:off'
			r = requests.get(url)
			#Save content of request response in file
			f = open(file_path,'wb')
			f.write(r.content)
			f.close()
			img = Image.open(file_path)
			img = img.convert('RGB')
			data = np.asarray(img)
			data = np.delete(data,np.s_[400:],0)
			final_img = Image.fromarray(data)
			final_img.save(file_path)
		except Exception as ex:
			#Write to log if exception occurs and continue requests
			logger('\n'+'Exception seen for '+file_name+': '+str(ex))
	else:
		logger('\n'+str(file_name)+' already exists')

def main():
	#read API key from API file
	f = open('API Key.txt','r')
	api_key = f.read()
	f.close()
	#contains GPS Locations for India in csv, extracted from shape file
	gps_coords_path = os.path.join(dataset_dir,'GPS_coordinates.csv')
	if os.path.exists(gps_coords_path):
		f = open(gps_coords_path,'r')
		line = f.readlines()
		for coords in line:
			if coords[-1] == '\n':
				coords = coords[:-1]
			point = coords.split(',')
			latitude = point[0]
			longitude = point[1]
			get_img(latitude,longitude,api_key)
	
if __name__ == '__main__':
	main()
	
