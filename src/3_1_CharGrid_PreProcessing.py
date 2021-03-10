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
from classes.headerElement import HeaderElement
from classes.listItems import ItemList
from classes.listElement import ListElement
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import time

# has to be added to conf file
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

invoices_path = generator_path + "Invoices/"
png_path = generator_path + "PNG/"
chargrid_gt = gt_decoder_path + "x_chargrid/"

if not os.path.exists(chargrid_gt):
    os.makedirs(chargrid_gt)

annotations_file = 'PDF_Annotations_' + companies_version + '_' + str(start_pdf) + "_" + str(end_pdf) + '.pkl'
annotations = pickle.load(open(generator_path + annotations_file, 'rb'))

start_time = time.time()

Chars = []
CharsId = []

with open(chargrid_mapping_path, encoding="utf-8") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        Chars.append(row[0])
        CharsId.append(row[1])

# iterate through all documents 
for N in range(start_pdf, end_pdf): # len(meta)
    if(N%10 == 0):
        spend = time.time() - start_time
        print("Parsing: ", N, " seconds: ", spend)

    # Load Documents and meta information
    currentDocument = annotations[N]
    id = currentDocument['ID']

    # Tessaract
    image = Image.open(png_path + str(id) + ".png")
    w, h = image.size
    characters = pytesseract.image_to_boxes(image).splitlines()

    # CharGrid
    CharGridInput = np.zeros((height, width, 50), dtype=int)
    
    for charInfo in characters:
        char = charInfo.split(" ")
        c = char[0].upper()
        if c in Chars:
            index = Chars.index(c)
            idC = int(CharsId[index])
            vec = np.zeros((50,), dtype=int)
            vec[idC] = 1
            cx = int((float(char[1])/4)*0.43)
            cy = height-int((float(char[4])/4)*0.43)
            cx2 = int((float(char[3])/4)*0.43)
            cy2 = height-int((float(char[2])/4)*0.43)
            for xVAL in range(cx, cx2):
                for yVAL in range(cy, cy2):
                    CharGridInput[yVAL,xVAL] = vec

    # plt.imshow(CharGridInput[:,:,1])
    # plt.show()

    np.save(chargrid_gt + str(id) +".npy", CharGridInput)

print("--- %s seconds ---" % (time.time() - start_time))
