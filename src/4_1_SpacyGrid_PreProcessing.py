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
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import spacy
import time

Visualization = True

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

configFile = configparser.ConfigParser()
configFile.read('settings.ini')
config = configFile['DEFAULT']

# Settings
companies_version = config['CompaniesVersion']
start_pdf = int(config['StartPDF'])
end_pdf = int(config['EndPDF'])

tesseract_conf_threshold = int(config['TesseractThreshold'])
width = int(config['ModelWidth'])
height = int(config['ModelHeight'])

generator_path = config['GeneratorPath']
gt_decoder_path = config['GroundTruth']
chargrid_mapping_path = config['ChargridMapping']
spacy_model_name = config['SpacyModel']

model = spacy.load(spacy_model_name)

invoices_path = generator_path + "Invoices/"
png_path = generator_path + "PNG/"
spacy_gt = gt_decoder_path + "x_spacy/"

if not os.path.exists(spacy_gt):
    os.makedirs(spacy_gt)

annotations_file = 'PDF_Annotations_' + companies_version + '_' + str(start_pdf) + "_" + str(end_pdf) + '.pkl'
annotations = pickle.load(open(generator_path + annotations_file, 'rb'))

start_time = time.time()

broken = 0
for N in range(start_pdf, end_pdf):
    if(N%10 == 0):
        spend = time.time() - start_time
        print("Parsing: ", N, " seconds: ", spend, "broken: ", broken)

    # Load Documents and meta information
    currentDocument = annotations[N]
    id = currentDocument['ID']

    # Tessaract
    image = Image.open(png_path + str(id) + ".png")
    w, h = image.size
    words = pytesseract.image_to_data(image, config="", output_type=pytesseract.Output.DATAFRAME, pandas_config=None)
    words = words[words['conf']>tesseract_conf_threshold]

    # text = words["text"]
    completePDF = ""
    for index, word in words.iterrows():
        completePDF += str(word["text"]) + " "

    SpacyGrid = np.zeros((height, width, 96), dtype=int)
    embeddings = model(completePDF)

    i = 0
    brokenDoc = False
    for index, word in words.iterrows():
        current = ""
        elements = 1
        vector = np.zeros(96)
        x = int((float(word["left"])/4)*0.43)
        y = int((float(word["top"])/4)*0.43)
        w = int((float(word["width"])/4)*0.43)
        h = int((float(word["height"])/4)*0.43)
        text = str(word["text"])
        while text != current or i > len(embeddings):
            current += embeddings[i].text
            vector += embeddings[i].vector
            elements += 1
            i +=1
        
        vector = vector / elements
        if word["text"] != current:
            brokenDoc = True
            broken = broken + 1
        for xc in range(x, x+w):
            for yc in range(y, y+h):
                SpacyGrid[yc, xc] = vector

    np.save(spacy_gt + str(id) +".npy", SpacyGrid)

print("--- %s seconds ---" % (time.time() - start_time))