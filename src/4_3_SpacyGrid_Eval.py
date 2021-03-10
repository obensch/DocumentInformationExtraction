import numpy as np
import math
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import keras
from keras.models import load_model

modelFile = ""
Start = 11000
batch_size = 50
num_batches = 1
num_classes = 9
num_anchors = 6
channels = 96
model_type = "seg" # "coord" "box" "seg"

calculate_MSE = False

parser_path = "data/3_Parser/"
data_path = "data/5_2_SpacyGrid_Data/"

model = load_model(data_path + "models/" + modelFile + ".md")

for off in range(num_batches):
    cur_batch = Start + (off*batch_size)
    print("Predicting: ", cur_batch )
    batch_X_test = np.zeros((batch_size, 362, 256, channels))
    for i in range(batch_size):
        cur_id = Start + i + (off*batch_size)
        x = np.load("data/5_1_SpacyGrid_PreProcessing/x/" + str(cur_id)+".npy")
        batch_X_test[i] = x

    if model_type == "box":
        pred_seg, pred_box = model.predict(batch_X_test)
    if model_type == "coord":
        pred_seg, pred_box, pred_coord = model.predict(batch_X_test)
    else:
        pred_seg = model.predict(batch_X_test)