# -*- coding: utf-8 -*-
"""
Created on Sat May  9 11:57:42 2020

@author: 91782

from sklearn import Ridge
"""
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

def read_csv(csv_path):
	ntl_wealth_data = pd.read_csv(csv_path)
	scatter_plot(ntl_wealth_data,'NTL_Intensity', 'Wealth_index')

def scatter_plot(data,feature, target):
	plt.figure(figsize=(16,8))
	plt.scatter(data[feature],data[target],c='black')
	plt.xlabel("Night Light Intensity")
	plt.ylabel("Wealth Index")
	plt.show()
	
#Get feature vector as list
def get_proper_featurevector(dataframe_feature):
    x1 = dataframe_feature
    values_x1 = x1.split(",")
    zero = values_x1[0].split("[")
    floatzero = float(zero[1])
    last = values_x1 [-1].split("]")
    floatlast = float(last[0])
    fvector = [float(i) for i in values_x1 [1:-1]]
    fvector.insert(0,floatzero)
    fvector.append(floatlast)
    return fvector

def makeArray(text):
    return np.fromstring(text,sep=', ')
def ridge_reg():
	#Feature from last layer of CNN are stored in CSV
	feature_embedding = pd.read_csv('E:\Datasets\DHS_NTL_FV.csv')
	#Wealth Index for GPS coordinates are stored in CSV
	gps_wealth = pd.read_csv('E:\\Datasets\\final_gps_wealth_ntl.csv')
	column_names = ["Features", "Wealth"]
	reg_df = pd.DataFrame(columns = column_names)
	if not os.path.exists('E:\\Datasets\\feature_weath.csv'):
		f = open('E:\\Datasets\\feature_weath.csv', 'a')
		for x in range(len(feature_embedding)):
			try:
				print(x)
				#Get feature vector as a vector
				feature_vector = get_proper_featurevector(feature_embedding['Feature'][x])
				#Check all indices where latitude value matches
				indices= [i for i,lat in enumerate(gps_wealth['Latitude']) if lat == feature_embedding['Latitude'][x]]
				for j in indices:
					#If Latitude value matches, check if longitude matches as well
					if float(feature_embedding['Longitude'][x][:-4]) == gps_wealth['Longitude'][j]:
						#If GPS coordinates match, store the wealth index for feature vector
						wealth_index = gps_wealth['Wealth'][j]
						break
				#Create data frame with one row and add it to csv
				reg_df.loc[0] = [feature_vector,wealth_index]
				reg_df.to_csv(f, header=False)
			except Exception as ex:
				#Exception handling is done to continue processing even if some values cause errors
				continue
		f.close()
	else:
		pass
	np.random.seed(123)
	#Read the csv with mapping from feature vectors to wealth index
	feature_wealth  = pd.read_csv('E:\\Datasets\\feature_weath.csv')
	feature_wealth.columns = ['Index','Features','Wealth']
	feature_wealth = feature_wealth.drop(['Index'],axis=1)
	feature_wealth['Features'] = feature_wealth['Features'].apply(makeArray)

	#Define multiple alpha values to find best match
	alphas_list = np.logspace(-1, 5, 7)
	final = []
	for alpha in alphas_list:
		try:
			#Split to test and training Data
			X_train,X_test,y_train,y_test = train_test_split(feature_wealth['Features'].as_matrix(),feature_wealth['Wealth'].as_matrix(),test_size=0.3,random_state=3)
			reg = Ridge(alpha=alpha)
			#Fit using Ridge regression object
			reg.fit(X_train, y_train)
			#Calculate Regression score for comparison
			s = reg.score(X_test,y_test)
			scores.append(s)
			final.append(np.mean(scores))
			print('R^2 of the best model: {:.3f}'.format(np.max(final)))
		except Exception as ex:
			print(ex)

def main():
	csv_path = 'E:\\Datasets\\ntl_wealth_1.csv'
	#read_csv(csv_path)
	ridge_reg()
	
if __name__ == '__main__':
	main()