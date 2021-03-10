import configparser
import os, glob
import fitz
import pickle
import pytesseract
import random
import numpy as np
import scipy.ndimage
import csv
import matplotlib.pyplot as plt
import cv2
from classes.headerElement import HeaderElement
from classes.listItems import ItemList
from classes.listElement import ListElement
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import time

configFile = configparser.ConfigParser()
configFile.read('settings.ini')
config = configFile['DEFAULT']

# Settings
companies_version = config['CompaniesVersion']
start_pdf = int(config['StartPDF'])
end_pdf = int(config['EndPDF'])

width = int(config['ModelWidth'])
height = int(config['ModelHeight'])

generator_path = config['GeneratorPath']
gt_decoder_path = config['GroundTruth']
invoices_path = generator_path + "Invoices/"

png_path = generator_path + "PNG/"
seg_path = gt_decoder_path + "seg/"
box_path = gt_decoder_path + "box/"
coord_path = gt_decoder_path + "coord/"

if not os.path.exists(png_path):
    os.makedirs(png_path)
if not os.path.exists(seg_path):
    os.makedirs(seg_path)
if not os.path.exists(box_path):
    os.makedirs(box_path)
if not os.path.exists(coord_path):
    os.makedirs(coord_path)

# load company data
annotations_file = 'PDF_Annotations_' + companies_version + '_' + str(start_pdf) + "_" + str(end_pdf) + '.pkl'
annotations = pickle.load(open(generator_path + annotations_file, 'rb'))

start_time = time.time()

# Semantic classes
num_classes = 9 # 1 Background class, 5 Header classes, 3 List classes
BackgroundVec =     [1, 0, 0, 0, 0, 0, 0, 0, 0] # 0
NameVec =           [0, 1, 0, 0, 0, 0, 0, 0, 0] # 1
NumberVec =         [0, 0, 1, 0, 0, 0, 0, 0, 0] # 2
DateVec =           [0, 0, 0, 1, 0, 0, 0, 0, 0] # 3
AmountVec =         [0, 0, 0, 0, 1, 0, 0, 0, 0] # 4
AddressVec =        [0, 0, 0, 0, 0, 1, 0, 0, 0] # 5
ItemNameVec =       [0, 0, 0, 0, 0, 0, 1, 0, 0] # 6
ItemQuantityVec =   [0, 0, 0, 0, 0, 0, 0, 1, 0] # 7
ItemAmountVec =     [0, 0, 0, 0, 0, 0, 0, 0, 1] # 8

# BoxMask classes
num_anchors = 6 # 5 Header classes, 1 List class

# return Id of every box
def boxMaskMapper(id):
    if id == "Name":
        return 0
    elif id == "Number":
        return 1
    elif id == "Date":
        return 2
    elif id == "Amount":
        return 3
    elif id == "Street" or id == "PostCode" or id == "City":
        return 4

# return corresponding vector for each field
def segMapper(id):
    if id == "Background":
        return BackgroundVec
    elif id == "Name":
        return NameVec
    elif id == "Number":
        return NumberVec
    elif id == "Date":
        return DateVec
    elif id == "Amount":
        return AmountVec
    elif id == "Street" or id == "PostCode" or id == "City":
        return AddressVec
    elif id == "ItemName":
        return ItemNameVec
    elif id == "ItemQuantity":
        return ItemQuantityVec
    elif id == "ItemAmount":
        return ItemAmountVec
    else:
        print("Mapping Error!")

def getBounds(currentField):
    # get bounds from a field
    cx = int(float(currentField.startX)*0.43)
    cy = int(float(currentField.endY)*0.43)
    cx2 = int(float(currentField.endX)*0.43)
    cy2 = int(float(currentField.startY)*0.43)
    return cx, cy, cx2, cy2 

def getCoords(currentField):
    # get coordinates from a field
    cx = int(float(currentField.startX)*0.43)
    cy = height-int(float(currentField.endY)*0.43)
    cx2 = int(float(currentField.endX)*0.43)
    cy2 = height-int(float(currentField.startY)*0.43)
    w = cx2 - cx
    h = cy2 - cy
    x = cx + (w / 2)
    y = cy + (h / 2)
    return x, y, w, h

# loop to parse every document
for N in range(start_pdf, end_pdf):
    if(N%500 == 0):
        spend = time.time() - start_time
        print("Parsing: ", N, " seconds: ", spend)
    
    # Load Documents and meta information
    currentDocument = annotations[N]
    id = currentDocument['ID']
    headerFields = currentDocument["HeaderElements"]
    ListElements = currentDocument["ListElements"]

    # load document
    doc = fitz.open(invoices_path + str(id) + ".pdf")
    page = doc[0]

    # convert to png
    zoom = 4
    mat = fitz.Matrix(zoom, zoom)
    image = page.getPixmap(matrix = mat, alpha = False) 
    w = image.width
    h = image.height
    image.writePNG(png_path + str(id) + ".png")

    # Create training data
    SegmentImage = np.zeros((height-3, width-1, num_classes), dtype=int)
    BoxMasks = np.zeros((height-3, width-1, 2*num_anchors), dtype=int)
    BoxCoords = np.zeros((height-3, width-1, 4*num_anchors), dtype=int)

    # Fill Seg Mapper with background
    for x in range(width-1):
        for y in range(height-3):
            SegmentImage[y,x] = segMapper("Background")
    
    # invert every second boxmask
    for ancs in range(0,num_anchors):
        BoxMasks[:,:,(ancs*2)+1] = np.ones((height-3,width-1), dtype=int)

    # Fill header fields in arrays
    for i in range(0, len(headerFields)):
        currentField = headerFields[i]
        cx, cy, cx2, cy2 = getBounds(currentField)
        x, y, w, h = getCoords(currentField)
        # iterate from left to right and top to bot
        for xVAL in range(cx, cx2):
            for yVAL in range(cy2, cy):
                SegmentImage[yVAL,xVAL] = segMapper(currentField.id)

                idBoxMask = boxMaskMapper(currentField.id)
                BoxMasks[yVAL,xVAL,idBoxMask*2] = 1 
                BoxMasks[yVAL,xVAL,(idBoxMask*2)+1] = 0 

                BoxCoords[yVAL,xVAL,idBoxMask*4] = x 
                BoxCoords[yVAL,xVAL,(idBoxMask*4)+1] = y 
                BoxCoords[yVAL,xVAL,(idBoxMask*4)+2] = w 
                BoxCoords[yVAL,xVAL,(idBoxMask*4)+3] = h 
    
    # iterate through every list element
    for i in range(0, len(ListElements)):
        currentField = ListElements[i]
        cx, cy, cx2, cy2 = getBounds(currentField)
        x, y, w, h = getCoords(currentField)
        w = width / 2
        # fill segment boxes 
        for xVAL in range(cx, cx2):
            for yVAL in range(cy2, cy):
                SegmentImage[yVAL,xVAL] = segMapper(currentField.id)

        # Fill box masks and box coords for list items once per row
        if currentField.id == "ItemName":
            for xVal in range(0,width-1):
                for yVal in range(cy2, cy):
                    BoxMasks[yVal,xVal,(num_anchors-1)*2] = 1 
                    BoxMasks[yVal,xVal,((num_anchors-1)*2)+1] = 0 

                    BoxCoords[yVal,xVal,(num_anchors-1)*4] = w # center x is for list items also w 
                    BoxCoords[yVal,xVal,((num_anchors-1)*4)+1] = y 
                    BoxCoords[yVal,xVal,((num_anchors-1)*4)+2] = w 
                    BoxCoords[yVal,xVal,((num_anchors-1)*4)+3] = h 

    # save target data
    np.save(seg_path + str(id) +".npy", SegmentImage)
    np.save(box_path + str(id) +".npy", BoxMasks)
    np.save(coord_path + str(id) +".npy", BoxCoords)

print("--- %s seconds ---" % (time.time() - start_time))