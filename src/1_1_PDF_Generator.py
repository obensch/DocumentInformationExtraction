import configparser
import os, glob
import fitz
import pickle
import csv
from classes.headerElement import HeaderElement
from classes.listItems import ItemList
from classes.listElement import ListElement
import classes.fakeGenerator as fakeGenerator
import random
import time

configFile = configparser.ConfigParser()
configFile.read('settings.ini')
config = configFile['DEFAULT']

# Settings
companies_version = config['CompaniesVersion']
number_companies = int(config['NumberOfCompanies'])
number_of_templates = int(config['TemplatesAmount'])
start_pdf = int(config['StartPDF'])
end_pdf = int(config['EndPDF'])

templatePath = config['TemplatesFolder']

generator_path = config['GeneratorPath']
invoicesPath = generator_path + "Invoices/"

if not os.path.exists(invoicesPath):
    os.makedirs(invoicesPath)

# load company data
companies_meta_file = "Companies_Meta_" + companies_version + "_" + str(number_companies) + ".pkl"
companies = pickle.load(open(generator_path + companies_meta_file, 'rb'))

start_time = time.time()
debug = config['Debug']

def load_random_template():
    """
    Load a random template from the given list.
    """
    TemplateHeader = []
    TemplateOther = []
    TItems = ItemList()
    RandomTemplate = random.randint(1,number_of_templates)
    with open(templatePath + str(RandomTemplate) + '.csv', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == "Header":
                element = HeaderElement(row[1])
                element.pdfX = int(row[2])
                element.pdfY = int(row[3])
                element.fontSize = int(row[4])
                TemplateHeader.append(element)
            elif row[0] == "Items":
                TItems.pdfX = int(row[1])
                TItems.pdfY = int(row[2])
                TItems.fontSize = int(row[3])
                TItems.xOffset = int(row[4])
                TItems.yOffset = int(row[5])
            elif row[0] == "Other":
                element = HeaderElement(row[1])
                element.pdfX = int(row[2])
                element.pdfY = int(row[3])
                element.fontSize = int(row[4])
                TemplateOther.append(element)
    return TemplateHeader, TemplateOther, TItems, RandomTemplate

def insert(page, pdfx, pdfy, content, size, font="tiro"):
    page.insertText(fitz.Point(pdfx, pdfy), content, fontname=font, fontsize=size)
    return page

missingBox = 0
Meta = []
currentID = 0
# Loop to create the PDF documents  
for i in range(start_pdf, end_pdf):
    TemplateHeader, TemplateOther, TItems, RTID = load_random_template()
    if(i%500 == 0):
        spend = time.time() - start_time
        print("Parsing: ", i, " Files missing one or more meta-infos: ", missingBox, " seconds: ", spend)

    # Create new PDF and select first page
    doc = fitz.open() 
    page = doc.newPage()  
    fonts = ["tiro"]
    font = random.choice(fonts)
    # page.insertFont(fontname="arial", fontfile="data/2_PDFGenerator/Fonts/arial.ttf") 

    if debug:
        page = insert(page, 12, 12, "T:" + str(RTID), 10)

    # load current company and its items
    company = companies[i]
    items = company["Items"]

    # set a random offset
    offsetX = random.randint(0,5)
    offsetY = random.randint(0,15)

    # add header elements
    HeaderElementsMeta = []
    for e in range(0, len(TemplateHeader)):
        # build HeaderElement
        cE = HeaderElement(TemplateHeader[e].id)
        cE.content = company[TemplateHeader[e].id]
        cE.pdfX = TemplateHeader[e].pdfX + offsetX
        cE.pdfY = TemplateHeader[e].pdfY + offsetY
        cE.fontName = TemplateHeader[e].fontName
        cE.fontSize = TemplateHeader[e].fontSize
        
        page = insert(page, cE.pdfX, cE.pdfY, cE.content, cE.fontSize) # add header element to PDF
        HeaderElementsMeta.append(cE) # add Meta

    # add list item elements
    ListElementsMeta = []
    for it in range(0, len(items)):
        # select current item
        curItem = items[it]
        # caluclate y-offset for each item
        yPos = TItems.pdfY + (it*TItems.yOffset) + offsetY

        # Randomly add a table header
        tableHeader = ["Beschreibung", "Menge", "Gesamt"]  
        if random.randint(0,1) == 0 and it == 0:
            for tI, tHeader in enumerate(tableHeader):
                cX = TItems.pdfX + (tI*TItems.xOffset) + offsetX
                cY = yPos - TItems.yOffset
                page = insert(page, cX, cY, tHeader, TItems.fontSize+2)

        # create ListElement name object (meta information)
        listElements = ["ItemName", "ItemQuantity", "ItemAmount"]
        curIt = []
        for lIndex, listElement in enumerate(listElements):
            cE = ListElement(listElement)
            cE.itemNumber = it
            cE.content = str(curItem[listElement])
            cE.pdfX = TItems.pdfX + (lIndex*TItems.xOffset) + offsetX
            cE.pdfY = yPos 
            cE.fontSize = TItems.fontSize
            curIt.append(cE)

        # add elements to PDF / add meta info to array
        for cI in range(0,len(curIt)):
            page = insert(page, curIt[cI].pdfX, curIt[cI].pdfY, curIt[cI].content, curIt[cI].fontSize)
            ListElementsMeta.append(curIt[cI]) # add meta info to array

    # add other fields to invoice
    for e in range(0, len(TemplateOther)):
        cE = TemplateOther[e]
        if cE.id == 'InvoiceLabel':
            page = insert(page, cE.pdfX + offsetX, cE.pdfY + offsetY, "Rechnung", cE.fontSize)
        elif cE.id == 'Rec':
            page = insert(page, cE.pdfX + offsetX, cE.pdfY + offsetY, "Empfänger", cE.fontSize)
        elif cE.id == 'KDN':
            kdn = fakeGenerator.KDN()
            page = insert(page, cE.pdfX + offsetX, cE.pdfY + offsetY, kdn, cE.fontSize)
        elif cE.id == 'InvoiceTopText':
            topText = fakeGenerator.topText()
            page = insert(page, cE.pdfX + offsetX, cE.pdfY + offsetY, topText, cE.fontSize)
        elif cE.id == 'InvoiceBotText':
            botText = fakeGenerator.botText()
            page = insert(page, cE.pdfX + offsetX, cE.pdfY + offsetY, botText, cE.fontSize)
        elif cE.id == 'Contact':
            contactBlock = fakeGenerator.contactBlock()
            page = insert(page, cE.pdfX + offsetX, cE.pdfY + offsetY, contactBlock, cE.fontSize)
        elif cE.id == 'NameBot':
            nameBot = fakeGenerator.nameBot()
            page = insert(page, cE.pdfX + offsetX, cE.pdfY + offsetY, nameBot, cE.fontSize)  
        elif cE.id == 'ContactBot':
            contactBot = fakeGenerator.contactBot()
            page = insert(page, cE.pdfX + offsetX, cE.pdfY + offsetY, contactBot, cE.fontSize)    
        elif cE.id == 'Bank':
            bank = fakeGenerator.bank()
            page = insert(page, cE.pdfX + offsetX, cE.pdfY + offsetY, bank, cE.fontSize)  
        elif cE.id == 'Tax':
            tax = fakeGenerator.taxOffice()
            page = insert(page, cE.pdfX + offsetX, cE.pdfY + offsetY, tax, cE.fontSize)  
        # Add non text items like a horizontal line
        # elif cE.id == 'LineH':
        #     start = fitz.Point(cE.pdfX, cE.pdfY)
        #     end = fitz.Point(page.MediaBoxSize[0]-cE.pdfX, cE.pdfY)
        #     page.drawLine(start, end)
        else:
            print("Template error for field:" + cE.content + " template: " + str(RTID))

    # load PyMuPDF bounding boxes
    pageDict = page.getText("dict")
    pageBlocks = pageDict["blocks"]

    # match PyMuPDF bouding boxes with added elements and store meta information about them
    for d in range(0, len(pageBlocks)):
        currentBlock = pageBlocks[d]
        lines = currentBlock["lines"]
        for l in range(0, len(lines)):
            currentLine = lines[l]
            spans = currentLine["spans"]             
            curElement = spans[0]
            origin = curElement["origin"]
            bbox = curElement["bbox"]
            content = curElement["text"]
            for e in range(0, len(HeaderElementsMeta)):
                if (content == HeaderElementsMeta[e].content) and (origin[0] == HeaderElementsMeta[e].pdfX) and origin[1] == HeaderElementsMeta[e].pdfY:
                    HeaderElementsMeta[e].startX = bbox[0]
                    HeaderElementsMeta[e].startY = bbox[1]
                    HeaderElementsMeta[e].endX = bbox[2]
                    HeaderElementsMeta[e].endY = bbox[3]
            for e in range(0, len(ListElementsMeta)):
                if (content == ListElementsMeta[e].content) and (origin[0] == ListElementsMeta[e].pdfX) and origin[1] == ListElementsMeta[e].pdfY:
                    ListElementsMeta[e].startX = bbox[0]
                    ListElementsMeta[e].startY = bbox[1]
                    ListElementsMeta[e].endX = bbox[2]
                    ListElementsMeta[e].endY = bbox[3]

    # Check if all fields have meta information
    brokenFile = False
    # Check header fields
    for e in range(0, len(HeaderElementsMeta)):
        if HeaderElementsMeta[e].startX == None:
            print("File: ", i, " Field: ", HeaderElementsMeta[e].content, " broken")
            brokenFile = True
    # Check item fields
    for e in range(0, len(ListElementsMeta)):
        if ListElementsMeta[e].startX == None:
            # print("File: ", i, " Field: ", ListElementsMeta[e].content, " broken")
            brokenFile = True
            
    # Only save file if meta information could be created
    if brokenFile:
        missingBox +=1
        doc.save(invoicesPath + str(currentID) + ".pdf")
        currentID = currentID + 1
    else:
        invoiceMeta = {
            "ID": currentID,
            "ListItemsNumber": len(items),
            "HeaderElements": HeaderElementsMeta,
            "ListElements": ListElementsMeta,
        }
        Meta.append(invoiceMeta)
        doc.save(invoicesPath + str(currentID) + ".pdf")
        currentID = currentID + 1

print("MissingBoxes: ", missingBox)

MetaInfoFileName = 'PDF_Annotations_' + companies_version + '_' + str(start_pdf) + "_" + str(end_pdf) + '.pkl'
pickle.dump(Meta, open(generator_path + MetaInfoFileName, 'wb'))

print("--- %s seconds ---" % (time.time() - start_time))