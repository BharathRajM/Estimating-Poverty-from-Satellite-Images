#!/usr/bin/env python
# coding: utf-8
'''
@author:Bharath
'''

from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from imutils import paths
import tensorflow
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras import Input
import argparse


ap = argparse.ArgumentParser()

ap.add_argument("-m","--model", required = True, help="path to model")
ap.add_argument("-i","--imageDirectory", required = True, help="path to image directory")
ap.add_argument("-o","--outputFilename",required = True, help = "output feature vectors file name")
args = vars(ap.parse_args())

modelPath = args["model"]

model = load_model(modelPath)
print("INFO] loading network...")
print(model.summary())

#extracting the feature embeddings of the last but 1 convulution layer (output size = 2*2*4096)
extract = Model(model.inputs,model.layers[-5].output)

#images = list(paths.list_images("Google_Imgs/"))
images = sorted(list(paths.list_images(args["imageDirectory"])))

df = pd.DataFrame(columns=["Latitude","Longitude","PredictedNTL","Feature"])

def get_feature_embedding(image):
    image = cv2.imread(image)
    if image is not None:
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        #this is the feature vector for the image
        features = extract.predict(image)
        #print(features.shape)
        features = features.reshape(features.shape[0], 2 * 2 * 4096)
        predicted_intensity = model.predict(image)
    else:
        features = None
        predicted_intensity = "Unknown"
    return features,predicted_intensity

#if memory issues are present, then you can change the range to a smaller value below and iterate this repeatedly
for x in range(0,2):#len(images)):
    feature,predicted_intensity = get_feature_embedding(images[x])
    #print(feature,"\n",predicted_intensity)
    if (feature is None):
        print("Found unknown:",images[x])
        continue
    else:
        latitude = str(images[x]).split("/")[1].split("_")[0]
        longitude = str(images[x]).split("/")[1].split("_")[1]
        final = list(np.array(feature[0]))
        df2 = {'Latitude': latitude, 'Longitude': longitude, 'PredictedNTL': predicted_intensity, 'Feature':final}
        df = df.append(df2, ignore_index=True)
        print(x)

df.to_csv(args["outputFilename"]+".csv")




