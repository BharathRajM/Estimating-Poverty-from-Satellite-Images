# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:15:15 2020

@author: 91782
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import csv
#Path for DHS Data file
dhs_file_path = 'E:\Datasets\IAHR74FL\IAHR74FL.dat'

def main():
	final_list=dict()
	with open(dhs_file_path,'r',encoding='utf-8') as dat_file:
		for line in dat_file:
			#Get cluster ID 
			cluster = int(line[15:23])
			'''
			Alterantive wealth index provided by DHS in 5 decimal format.
			wealth = float(line[239:245].strip())
			wealth = wealth/10000.0
			'''
			#Wealth index ranging from 1-5
			wealth = int(line[230:238].strip()[0])

			final_list[cluster]=[]
			final_list[cluster].append((wealth))
			
	reg_list = []
	for clusterid in final_list:
		#Aggregate all wealth indexs in each cluster
		num = sum(final_list[clusterid])
		den = len(final_list[clusterid])
		final_list[clusterid] = num/den
		reg_list.append([clusterid,final_list[clusterid]])
		#Write aggregated wealth index for each file into a csv
	with open('E:\Datasets\cluster_wealth.csv','w',newline='') as wealth_f:
		for i in range(0,len(reg_list)):
			cluster_ele = reg_list[i]
			wtr = csv.writer(wealth_f, quoting=csv.QUOTE_ALL)
			wtr.writerow(cluster_ele)
			
	print(reg_list)
	
if __name__ == '__main__':
	main()