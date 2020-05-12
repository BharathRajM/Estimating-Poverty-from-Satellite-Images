#!/usr/bin/env python
# coding: utf-8

# In[5]:


#to save figures in the backgroumd
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from VGGTransferNet import VGGTransferNet
from trainingmonitor import TrainingMonitor
from epochcheckpoint import EpochCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import os


# In[6]:


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True,
#	help="path to input dataset (i.e., directory of images)")

ap.add_argument("-c", "--checkpoint", required=True, help="path to output checkpoints directory")

ap.add_argument("-m", "--model", type=str,help="path to *specific* model checkpoint to load from")

ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output accuracy/loss plot")

ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")

args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 100
INIT_LR = 1e-6
BS = 32
IMAGE_DIMS = (400, 400, 3)

print("[INFO] loading images...")
random.seed(42)

# construct the image generator for data augmentation
datagen = ImageDataGenerator(rescale=1./255,rotation_range=25, vertical_flip=True, horizontal_flip=True, fill_mode="nearest")


train_it = datagen.flow_from_directory(directory = r"./Dataset/train/",target_size = (400,400), class_mode = "categorical",batch_size = BS, shuffle=True, seed=42)

valid_it = datagen.flow_from_directory(directory = r"./Dataset/test/",target_size = (400,400), class_mode = "categorical",batch_size = BS, shuffle=True, seed=42)


if args["model"] is None:
    # initialize the model
    print("[INFO] compiling model...")
    model = VGGTransferNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=3)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# otherwise, we're using a checkpoint model
else:
    # load the checkpoint from disk
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])
    
    # update the learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    
    #K.set_value(model.optimizer.lr, 1e-2)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

# build the path to the training plot and training history
plotPath = os.path.sep.join(["output", "Satellite_VGG.png"])
jsonPath = os.path.sep.join(["output", "Satellite_VGG.json"])
    
# construct the set of callbacks
callbacks = [EpochCheckpoint(args["checkpoint"], every=5,startAt=args["start_epoch"]), TrainingMonitor(plotPath,jsonPath=jsonPath,startAt=args["start_epoch"])]

STEP_SIZE_TRAIN=train_it.n//train_it.batch_size
STEP_SIZE_VALID=valid_it.n//valid_it.batch_size

# train the network
print("[INFO] training network...")
H = model.fit_generator( generator=train_it, validation_data=valid_it, steps_per_epoch=STEP_SIZE_TRAIN,validation_steps=STEP_SIZE_VALID, epochs=EPOCHS, verbose=1, callbacks=callbacks)

print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])

