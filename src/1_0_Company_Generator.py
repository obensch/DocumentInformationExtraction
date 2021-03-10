import configparser
import os, glob
from faker import Faker
import datetime
import random
import pickle

configFile = configparser.ConfigParser()
configFile.read('settings.ini')
config = configFile['DEFAULT']

# Settings
lang = config['FakerLocale']
version = config['CompaniesVersion']
number_companies = int(config['NumberOfCompanies'])
start_date = datetime.date(int(config['StartDateYear']), int(config['StartDateMonth']), int(config['StartDateDay']))
end_date = datetime.date(int(config['EndDateYear']), int(config['EndDateMonth']), int(config['EndDateDay']))

generator_path = config['GeneratorPath']
product_names_file = config['ProductNamesFile']

if not os.path.exists(generator_path):
    os.makedirs(generator_path)

def load_list_items():
    """ 
    Load all list item elements and return them as a list.
    Currently chemical elements from one file only
    """
    product_names = []
    elements_file = open(product_names_file, "r", encoding="utf8")
    for line in elements_file:
        product_names.append(line.rstrip())
    elements_file.close()
    return product_names

def generate_invoice_number(invoiceDate, invoiceName):
    """
    Generate a random invoice number by concatenating a random choice from each list.
    Extend by adding more lists, or extend the lists.
    """
    NumberSynonyms = ["Rechnung", "Rechnungsnummer", "RechnungsNr.", "Rechnung Nr.", "Nummer", "Nr.", "Rg.-Nr.", "Vorgangsnummer"]
    ColonOptions = [" ", ": "]
    StartOptions = ["", "RE-"]
    NameOptions = ["", invoiceName[0:random.randint(2, 5)].upper()+"-"]
    DateOptions = ["", invoiceDate.strftime("%Y-"), invoiceDate.strftime("%d-%m-%Y-"),  invoiceDate.strftime("%d/%m/%Y-")]
    invoiceNR = str(random.randrange(1, 99999)).zfill(5)
    # Concatenate the lists
    invoiceNumber = random.choice(NumberSynonyms) + random.choice(ColonOptions) + random.choice(StartOptions) + random.choice(NameOptions)
    invoiceNumber = invoiceNumber + random.choice(DateOptions) + invoiceNR
    return invoiceNumber

def generate_list_items(listItemList):
    """
    Generate the different list items by selecting a random element from each list.
    Needs an list of invoice names as an input.
    """
    listItems = []
    numberOfListItems = random.randint(1,10)
    currencies = [""," £", " $", " CHF"] # " €",
    totalSyn = ["", "Gesamt ", "Gesamt: ", "Total ", "Total: ", "Summe ", "Summe: ", "Endbertrag ", "Endbetrag: "]
    curr = random.choice(currencies)
    invoiceTotal = 0
    for x in range(0, numberOfListItems):
        itemName = random.choice(listItemList)
        itemQuantity = random.randint(1, 999)
        itemAmount = round(random.uniform(1.00, 999.00), 2)
        invoiceTotal += itemAmount
        if curr != " $":
            itemAmount = str(itemAmount).replace(".", ",")
        item = {
            "ItemName": itemName,
            "ItemQuantity": itemQuantity,
            "ItemAmount": str(itemAmount) + curr,
        }
        listItems.append(item)
    invoiceTotal = str("{:.2f}".format(invoiceTotal))
    if curr != " $":
        invoiceTotal = invoiceTotal.replace(".", ",")
    formatted_Total =  random.choice(totalSyn) + invoiceTotal + curr
    return listItems, formatted_Total

def generate_date(invoiceDate):
    """
    Generates a date string. Needs a python date as an input.
    """
    preSyn = ["Datum: ", "Datum ", "Rechnungsdatum: ", "Rechnungsdatum "]
    dateString = invoiceDate.strftime("%d-%m-%Y")
    concat = random.choice(preSyn) + dateString
    return concat

fake = Faker(lang)
Faker.seed(0)
listItemList = load_list_items() 

companies = []
# Generate company name for each company
company_names = [fake.unique.company() for i in range(number_companies)]
# Generate fake companies
for i in range(0, number_companies):
    if i%1000 == 0: 
        print("Processing:", i)
    invoiceName = company_names[i]

    # Generate a random address and split it into parts
    address = str(fake.address()).splitlines()
    street = address[0]
    postCode = address[1].split(" ")[0]
    city = address[1].split(" ")[1]

    # Generate random date, invoice number, date and list items
    invoiceDate = fake.date_between_dates(start_date, end_date)
    invoiceNumber = generate_invoice_number(invoiceDate, invoiceName) 
    listItems, invoiceTotal = generate_list_items(listItemList) 
    dateString = generate_date(invoiceDate)

    invoice = {
        "ID": i,
        "Name": invoiceName,
        "Number": invoiceNumber,
        "Date": dateString,
        "Street": street,
        "PostCode": postCode,
        "City": city,
        "Amount": invoiceTotal,
        "Items": listItems     
    }
    companies.append(invoice)

pickle.dump(companies, open(generator_path + 'Companies_Meta_' + version + "_" + str(number_companies) + '.pkl', 'wb'))