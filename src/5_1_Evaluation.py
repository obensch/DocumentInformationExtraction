import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import cv2
import pytesseract
from classes.headerElement import HeaderElement
from classes.listItems import ItemList
from classes.listElement import ListElement
import time
import os

start_time = time.time()

# Settings
startId = 11000
testSize = 1000
LayerStart = 1
LayerEnd = 9

# 0 BackgroundVec
# 1 Name
# 2 Number
# 3 Date 
# 4 Amount 
# 5 Address 
# 6 ItemName 
# 7 ItemQuantity
# 8 ItemAmount 

CharGridModel = ""
SpacyModel =""
Meta_Info_File = ""
threshold = 0.5
tesseract_conf_threshold = 10

parserPath = "data/3_Parser/"
generatorPath = "data/2_PDFGenerator/"
invoicePath = "Invoices/"
results_path = "data/6_Evaluation/Coord2/"

meta = pickle.load(open(generatorPath + Meta_Info_File, 'rb'))
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def calculateOverlap(box1,box2):    
    """    
    calculates the overlap of two boxes    
    returns the overlap relative to area of box1 and relative to area of box2 in one tuple, i.e. (overlap1,overlap2)    
    boxes should be lists like [xmin, ymin, xmax, ymax]     """    
    area_box1 = (box1[2]-box1[0])*(box1[3]-box1[1])    
    area_box2 = (box2[2]-box2[0])*(box2[3]-box2[1])    
    dx = min(box1[2], box2[2]) - max(box1[0], box2[0])    
    dy = min(box1[3], box2[3]) - max(box1[1], box2[1])    
    if (dx>=0) and (dy>=0):        
        return dx*dy/area_box1, dx*dy/area_box2    
    else:        
        return 0,0

def result_preprocess(np_array, current_layer):
    """
    select a layer, apply the threshold, upscale and find the contours afterwards
    return the upscaled layer and the contours
    """
    selected_layer = np_array[:,:,current_layer]
    adjust_size = np.zeros((362,256))
    adjust_size[:-3,:-1] = selected_layer
    upscale_layer = cv2.resize(adjust_size, dsize=(2380, 3368), interpolation=cv2.INTER_NEAREST)
    threshold_applied = np.where(upscale_layer < threshold, 0, 255).astype("uint8")
    contours, hierarchy = cv2.findContours(threshold_applied, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return upscale_layer, contours

def layer_mapping(layer_id):
    layer_array = ["Background", "Name", "Number", "Date", "Amount", "Address", "ItemName", "ItemQuantity", "ItemAmount"]
    return layer_array[layer_id]

def best_contour(layer, contours):
    """
    return bounding box of the contour with the highest mean
    """
    best_mean = 0
    best_bounds = [0,0,0,0]
    for cnt in cont_s:
        xT,yT,wT,hT = cv2.boundingRect(cnt)
        mask_mean = layer[(yT):(yT+hT), (xT):(xT+wT)].mean()
        if mask_mean > best_mean:
            best_mean = mask_mean
            best_bounds = [xT,yT,(xT+wT),(yT+hT)]
    return best_bounds

for current_layer in range(LayerStart, LayerEnd):
    GTM_GTI = 0
    GTI_RS = 0
    GTI_RC = 0
    brokeFiles = []
    for i in range(testSize):
        currId = startId + i
        # print("Evaluating Id: " + str(currId) + " broke:" + str(GTM_GTI) + " RC:" + str(GTI_RC) + " RS:" + str(GTI_RS))
        groundTruthMeta = meta[currId]
        h_fields = groundTruthMeta["HeaderElements"]
        l_elements = groundTruthMeta["ListElements"]

        layer_name = layer_mapping(current_layer)

        if current_layer < 5:
            for h_field in h_fields:
                if h_field.id == layer_name:
                    GTM = h_field.content

        if current_layer == 5:
            for h_field in h_fields:
                if h_field.id == 'Street':
                    GTM = h_field.content
            for h_field in h_fields:
                if h_field.id == 'PostCode' or h_field.id == 'City':
                    GTM = GTM + " " + h_field.content

        if current_layer >= 6:
            ListItemNames = []
            for LI in range(0, len(l_elements)):
                curLElement = l_elements[LI]
                if curLElement.id == layer_name:
                    ListItemNames.append(curLElement.content)
            GTM = " ".join(ListItemNames)

        GTM = GTM.replace("Rechnung  Nr.", "Rechnung Nr.")

        # load invoice images and run and extract all_words with pytessaract
        image = cv2.imread(parserPath + invoicePath + "/Invoice-" + str(currId) + ".png")
        all_Words = pytesseract.image_to_data(image, config="", output_type=pytesseract.Output.DATAFRAME, lang='deu', pandas_config=None)
        all_Words = all_Words[all_Words['conf']>tesseract_conf_threshold]
        # print(all_Words.head())

        # load GT data and SpacyGrid / CharGrid results
        gt_y = np.load("data/3_Parser/seg/" + str(currId)+".npy")
        cg_y = np.load("data/4_2_CharGrid_Data/results/" + CharGridModel + "/Seg_" + str(currId)+".npy")
        sg_y = np.load("data/5_2_SpacyGrid_Data/results/" + SpacyModel + "/Seg_" + str(currId)+".npy")

        up_t, cont_t = result_preprocess(gt_y, current_layer)
        up_c, cont_c = result_preprocess(cg_y, current_layer)
        up_s, cont_s = result_preprocess(sg_y, current_layer)
        
        if len(cont_t) != 1 and current_layer < 6 :
            print("ID: " + str(currId) + " broken")

        text_t = []
        text_c = []
        text_s = []

        # load image
        source_img = Image.open(parserPath + invoicePath + "/Invoice-" + str(currId) + ".png")
        draw = ImageDraw.Draw(source_img)

        for index, word in all_Words.iterrows():
            xW = word["left"]
            yW = word["top"]
            wW = word["width"]
            hW = word["height"]
            boundsWord = [xW,yW,(xW+wW),(yW+hW)]

            # shape = (((int(currentField.startX*zoom), int(currentField.startY*zoom)), (int(currentField.endX*zoom), int(currentField.endY*zoom))))
            draw.rectangle(boundsWord, outline ="red", width=5)

            if current_layer < 6:
                x,y,w,h = cv2.boundingRect(cont_t[0])
                box_t = [x,y,(x+w),(y+h)]
                box_c = best_contour(up_c, cont_c)
                box_s = best_contour(up_s, cont_s)
                o1_t, o2_t = calculateOverlap(box_t, boundsWord)
                o1_c, o2_c = calculateOverlap(box_c, boundsWord)
                o1_s, o2_s = calculateOverlap(box_s, boundsWord)
                # print("Word:", GTM, " o1: ", overlap1, " o2:", overlap2, " bWord:", word["text"])
                if(o2_t > threshold):
                    text_t.append(word["text"]) 
                if(o2_c > threshold):
                    text_c.append(word["text"]) 
                if(o2_s > threshold):
                    text_s.append(word["text"]) 
            else:
                for cnt in cont_t:
                    x,y,w,h = cv2.boundingRect(cnt)
                    box_t = [x,y,(x+w),(y+h)]
                    o1, o2 = calculateOverlap(box_t, boundsWord)
                    if(o2 > threshold):
                        text_t.append(str(word["text"])) 
                for cntC in cont_c:
                    x,y,w,h = cv2.boundingRect(cnt)
                    box_c = [x,y,(x+w),(y+h)]
                    o1, o2 = calculateOverlap(box_c, boundsWord)
                    if(o2 > threshold):
                        text_c.append(str(word["text"])) 
                for cnt in cont_s:
                    x,y,w,h = cv2.boundingRect(cnt)
                    box_s = [x,y,(x+w),(y+h)]
                    o1, o2 = calculateOverlap(box_s, boundsWord)
                    if(o2 > threshold):
                        text_s.append(str(word["text"])) 

        GTI = " ".join(text_t)
        GTC = " ".join(text_c)
        GTS = " ".join(text_s)
        spend = time.time() - start_time

        if GTMS != GTIS:
            GTM_GTI = GTM_GTI + 1
        if GTMS != GTSS:
            GTI_RS = GTI_RS + 1
        if GTMS != GTCS:
            GTI_RC = GTI_RC + 1

        if (GTMS != GTSS):
            brokeString = str(i) + "," + GTMS + "," + GTIS + "," + GTSS + "," + GTCS
            brokeFiles.append(brokeString)
            print("BROKEN ", brokeString) 
        
        if i%10 == 0 and i != 0:
            PERT = (GTM_GTI/i)*100
            PERS = (GTI_RS/i)*100
            PERC = (GTI_RC/i)*100
            print(str(i) + " Time:" + str(spend) + " brokeT:" + str(PERT) + "% brokeS:" + str(PERS) + "% brokeC:" + str(PERC) + "%")

    GTP = (GTM_GTI/testSize)*100
    RCP = (GTI_RC/testSize)*100
    RSP = (GTI_RS/testSize)*100
    resultsString = "Layer Id: " + str(current_layer) + " Broke: " + str(GTP) + "% Spacy: " + str(RSP) + "% Char: " + str(RCP) + "%"
    print(resultsString) 

    # if not os.path.exists(results_path):
    #     os.makedirs(results_path)

    # ResultsFile = str(current_layer) + ".txt"
    # with open(results_path + ResultsFile, 'w') as f:
    #     f.write(resultsString + "\n")
    #     for item in brokeFiles:
    #         f.write("%s\n" % item.encode("utf-8"))