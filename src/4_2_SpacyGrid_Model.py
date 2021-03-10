import numpy as np
import math
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Softmax, Input, InputLayer, ReLU
import random
# import pydotplus
# import pydot 
# from keras.utils.vis_utils import model_to_dot
# keras.utils.vis_utils.pydot = pydot
from keras.utils.vis_utils import plot_model
# from importlib import reload
# reload(keras.utils.vis_utils)


config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


configFile = configparser.ConfigParser()
configFile.read('settings.ini')
config = configFile['DEFAULT']

# Settings
companies_version = config['CompaniesVersion']
model_type = int(config['ModelType'])
num_classes = int(config['NumClasses'])
num_anchors = int(config['NumAnchors'])

width = int(config['ModelWidth'])
height = int(config['ModelHeight'])
batch_size_ini = int(config['BatchSize'])
num_train = int(config['NumTrain'])
num_val = int(config['NumVal'])
num_epochs = int(config['NumEpochs'])

generator_path = config['GeneratorPath']
gt_decoder_path = config['GroundTruth']
spacygrid_path = config['SpacygridPath']

if not os.path.exists(spacygrid_path):
    os.makedirs(spacygrid_path)

spacy_gt = gt_decoder_path + "x_spacy/"
seg_path = gt_decoder_path + "seg/"
box_path = gt_decoder_path + "box/"
coord_path = gt_decoder_path + "coord/"

channels = 96

total_array = []

for i in range(8000):
    total_array.append(f"{i}")

random.shuffle(total_array)

train_ids = total_array[:6000]
val_ids = total_array[6000:8000]

part_list = {'train': train_ids, 'validation': val_ids}

def add_block_a(inputL, channelsIn, channelsOut):
    conv1 = Conv2D(channelsIn, kernel_size=3, activation='relu', strides=2)(inputL)
    batch1 = BatchNormalization()(conv1)
    conv2 = Conv2D(channelsOut, kernel_size=3, activation='relu', padding="same")(batch1)
    batch2 = BatchNormalization()(conv2)
    conv3 = Conv2D(channelsOut, kernel_size=3, activation='relu', padding="same")(batch2)
    batch3 = BatchNormalization()(conv3)
    outBa = Dropout(0.1)(batch3)
    return outBa

def add_block_a_dash(inputL, channelsIn, channelsOut):
    conv1 = Conv2D(channelsIn, kernel_size=3, activation='relu', strides=2)(inputL)
    batch1 = BatchNormalization()(conv1)
    conv2 = Conv2D(channelsOut, kernel_size=3, activation='relu', padding="same", dilation_rate=1)(batch1)
    batch2 = BatchNormalization()(conv2)
    conv3 = Conv2D(channelsOut, kernel_size=3, activation='relu', padding="same", dilation_rate=1)(batch2)
    batch3 = BatchNormalization()(conv3)
    outBaD = Dropout(0.1)(batch3)
    return outBaD

def add_block_b(inputL, channelsIn, channelsOut):
    conv1 = Conv2D(channelsIn, kernel_size=1, activation='relu')(inputL)
    batch1 = BatchNormalization()(conv1)
    conv2 = Conv2DTranspose(channelsOut, kernel_size=3, activation='relu', strides=2)(batch1)
    batch2 = BatchNormalization()(conv2)
    conv3 = Conv2D(channelsOut, kernel_size=3, activation='relu', padding="same", dilation_rate=1)(batch2)
    batch3 = BatchNormalization()(conv3)
    conv4 = Conv2D(channelsOut, kernel_size=3, activation='relu', padding="same", dilation_rate=1)(batch3)
    batch4 = BatchNormalization()(conv4)
    outBb = Dropout(0.1)(batch4)
    return outBb

def add_block_c(inputL, channelsIn, channelsOut):
    conv1 = Conv2D(channelsIn, kernel_size=1, activation='relu')(inputL)
    batch1 = BatchNormalization()(conv1)
    conv2 = Conv2DTranspose(channelsOut, kernel_size=3, activation='relu', strides=2)(batch1)
    batch2 = BatchNormalization()(conv2)
    outBc = Dropout(0.1)(batch2)
    return outBc

def add_block_d_e(inputL, channelsIn, channelsOut):
    conv1 = Conv2D(channelsIn, kernel_size=3, activation='relu', padding="same")(inputL)
    batch1 = BatchNormalization()(conv1)
    conv2 = Conv2D(channelsIn, kernel_size=3, activation='relu', padding="same")(batch1)
    batch2 = BatchNormalization()(conv2)
    conv3 = Conv2DTranspose(channelsOut, kernel_size=3, activation='relu', strides=2)(batch2)
    batch3 = BatchNormalization()(conv3)
    outBd = Softmax()(batch3)
    return outBd

def add_block_f(inputL, channelsIn, channelsOut):
    conv1 = Conv2D(channelsIn, kernel_size=3, activation='relu', padding="same")(inputL)
    batch1 = BatchNormalization()(conv1)
    conv2 = Conv2D(channelsIn, kernel_size=3, activation='relu', padding="same")(batch1)
    batch2 = BatchNormalization()(conv2)
    conv3 = Conv2DTranspose(channelsOut, kernel_size=3, activation='relu', strides=2)(batch2)
    batch3 = BatchNormalization()(conv3)
    outBf = LeakyReLU()(batch3)
    return outBf

def build_network():
    inputL = Input(shape=(height, width, channels))
    # Encoder
    blockA = add_block_a(inputL, channels, 2*channels)          # first a block
    blockA2 = add_block_a(blockA, 2*channels,4*channels)        # second a block 
    blockAd = add_block_a_dash(blockA2, 4*channels,8*channels)  # a' block

    encoder  = Model(inputL, blockAd, name="Encoder")

    latentDim =Input(shape=(44, 31, 8*channels))

    # Semantic decoder
    blockB = add_block_b(latentDim, 8*channels, 4*channels)
    blockC = add_block_c(blockB, 4*channels, 2*channels)
    blockD = add_block_d_e(blockC, 2*channels, num_classes)

    segDecoder  = Model(latentDim, blockD, name="SegDecoder")

    # BBox decoder
    blockB = add_block_b(latentDim, 8*channels, 4*channels)
    blockC = add_block_c(blockB, 4*channels, 2*channels)
    blockE = add_block_d_e(blockC, 2*channels, 2*num_anchors)

    boxDecoder = Model(latentDim, blockE, name="BoxDecoder")

    # Coord decoder
    blockB = add_block_b(latentDim, 8*channels, 4*channels)
    blockC = add_block_c(blockB, 4*channels, 2*channels)
    blockF = add_block_f(blockC, 2*channels, 4*num_anchors)

    coordDecoder = Model(latentDim, blockF, name="CoordDecoder")

    outputDecoders=[]
    if model_type == 1:
        outputDecoders.append(segDecoder(encoder(inputL)))
    if model_type == 2:
        outputDecoders.append(segDecoder(encoder(inputL)))
        outputDecoders.append(boxDecoder(encoder(inputL)))
    if model_type == 3:
        outputDecoders.append(segDecoder(encoder(inputL)))
        outputDecoders.append(boxDecoder(encoder(inputL)))
        outputDecoders.append(coordDecoder(encoder(inputL)))

    autoencoder = Model(inputs=inputL, outputs=outputDecoders, name="CompleteModel")
    return autoencoder

class MyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size=5):
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.shuffle = False
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.total / self.batch_size)
    
    def __getitem__(self, idx):
        idxs = [i for i in range(idx*self.batch_size,(idx+1)*self.batch_size)]
        batch_X = np.zeros((self.batch_size, height, width, channels))
        batch_Y_Seg = np.zeros((self.batch_size, height-3, width-1, num_classes))
        batch_Y_Box = np.zeros((self.batch_size, height-3, width-1, 2*num_anchors))
        batch_Y_Coord = np.zeros((self.batch_size, height-3, width-1, 4*num_anchors))
        for i in range(self.batch_size):
            x = np.load(spacy_gt + str(idxs[i])+".npy")
            seg = np.load(seg_path + str(idxs[i])+".npy")
            box = np.load(box_path + str(idxs[i])+".npy")
            coord = np.load(coord_path + str(idxs[i])+".npy")
            batch_X[i] = x
            batch_Y_Seg[i] = seg
            batch_Y_Box[i] = box
            batch_Y_Coord[i] = coord
        
        if model_type == 1:
            y = batch_Y_Seg
        if model_type == 2:
            y = {"SegDecoder": batch_Y_Seg, "BoxDecoder": batch_Y_Box }
        if model_type == 3:
            y = {"SegDecoder": batch_Y_Seg, "BoxDecoder": batch_Y_Box, "CoordDecoder": batch_Y_Coord}
        return batch_X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

total = num_train + num_val

TrainGenerator = MyDataGenerator(start=0, end=num_train)
ValidGenerator = MyDataGenerator(start=num_train, end=total)

cp_folder = spacygrid_path + "models/" + model_type + "/CPs/" 
if not os.path.exists(cp_folder):
    os.makedirs(cp_folder)
checkpoint_path = cp_folder + companies_version + "_"+ str(total) +"I-{epoch:03d}E-{loss:03f}L-{val_loss:03f}VL.h5"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)

model = build_network()

if model_type == 1:
    losses = {"SegDecoder": "categorical_crossentropy"}
    lossWeights = {"SegDecoder": 1.0}
if model_type == 2:
    losses = {"SegDecoder": "categorical_crossentropy", "BoxDecoder": "binary_crossentropy",}
    lossWeights = {"SegDecoder": 1.0, "BoxDecoder": 1.0}
if model_type == 3:
    losses = {"SegDecoder": "categorical_crossentropy", "BoxDecoder": "binary_crossentropy", "CoordDecoder": "huber_loss",}
    lossWeights = {"SegDecoder": 1.0, "BoxDecoder": 1.0, "CoordDecoder": 1.0}

model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics=["acc"])

history = model.fit(TrainGenerator, validation_data=ValidGenerator, callbacks=[cp_callback], epochs=num_epochs) 
model.save(spacygrid_path + "models/" + model_type + "/" + companies_version + "_"+ str(total) +"I-" + str(num_epochs) + "E.md")

np.save(spacygrid_path + "models/" + model_type + "/" + companies_version + "_"+ str(total) +"I-" + str(num_epochs) + "E_metrics.npy", history.history)

print(history.history)
