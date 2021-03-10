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
num_batches = 20
num_classes = 9
num_anchors = 6
channels = 50
model_type = "coord" # "coord" "box" "seg"

calculate_MSE = False

parser_path = "data/3_Parser/"
data_path = "data/4_2_CharGrid_Data/"

model = load_model(data_path + "models/" + modelFile + ".md")

for off in range(num_batches):
    cur_batch = Start + (off*batch_size)
    print("Predicting: ", cur_batch )
    batch_X_test = np.zeros((batch_size, 362, 256, channels))
    for i in range(batch_size):
        cur_id = Start + i + (off*batch_size)
        x = np.load("data/4_1_CharGrid_PreProcessing/x/" + str(cur_id)+".npy")
        batch_X_test[i] = x

    if model_type == "box":
        pred_seg, pred_box = model.predict(batch_X_test)
    if model_type == "coord":
        pred_seg, pred_box, pred_coord = model.predict(batch_X_test)
    else:
        pred_seg = model.predict(batch_X_test)

    for i in range(batch_size):
        cur_id = Start + i + (off*batch_size)
        result_path = data_path + "/results/" + modelFile
        # create paths if needed
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        result_seg = pred_seg[i,:,:,:]
        np.save(result_path + "/Seg_" + str(cur_id) + ".npy", result_seg)
        if model_type == "box":
            result_box = pred_box[i,:,:,:]
            np.save(result_path + "/Box_" + str(cur_id) + ".npy", result_box)