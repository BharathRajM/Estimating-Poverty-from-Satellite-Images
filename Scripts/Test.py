# import the necessary packages
from keras.preprocessing.image import img_to_array
#from keras.models import load_model
#uncomment the above line and comment the line below if you are training for the batchsize8 model
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from imutils import paths

ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", type=str, required = True, help="path to *specific* model checkpoint to load from")

ap.add_argument("-t", "--testDir", required=True, help="path to input test images directory (i.e., directory of images)")

args = vars(ap.parse_args())

pathToTestImageDirectory = args["testDir"]

pathToModel = args["model"]

list_of_images = list(paths.list_images(pathToImage))

print("[INFO] loading network...")
model = load_model(pathToModel)


labels = ["Low Intensity","Medium Intensity","High Intensity"]

for pathToImage in list_of_images:
    image = cv2.imread(pathToImage)
    output = image.copy()
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    print(proba,pathToImage)
    idx = np.argmax(proba)
    print(labels[idx],pathToImage)
    label = "{}: {:.2f}% ".format(labels[idx], proba[idx] * 100)
    print(label,pathToImage)
    cv2.putText(output, label,(10,20), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imshow("Output", output)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    print("Next\n\n\n")
