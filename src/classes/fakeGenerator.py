from faker import Faker
import re
import random
import datetime

regex = re.compile('[^a-zßäöü]')

lang = "de_DE"
fake = Faker(lang)

topTextSyn = ["Vereinbarungsgemäß berechnen wir unsere Leistungen wie folgt:", "Wir bedanken uns für die gute Zusammenarbeit und stellen Ihnen verinbarungsgemäß folgende Lieferungen in Rechnung:"]
botTextSyn = ["Sofern nicht anders angegeben, entspricht das Liefer-/Leistungsdatum dem Rechnungsdatum. \nZahlungsbedingung netto zahlbar bis 13.06.2020"]
kdnSyn = ["Kundennummer", "KDN", "Kdn.", "KundenNr.", "KundenNr"]
bankSyn = ["Sparkasse Köln/Bonn", "Volksbank", "Postbank", "Deutsche Bank", "Commerzbank", "ING", "Santander"]

col = [" ", ": "]
webPre = ["", "www."]
webEnd = [".de",".com", ".net", ".eu"]
mailPre = ["info@", "invoice@", "invoicede@", "rechnung@"]

personPre = ["Kontakt", "Ansprechpartner"]
addressPre = ["Addresse", "Anschrift"]
telPre = ["Tel.", "Telefon"]
webStringPre = ["Web", "Website", "Internet"]
mailStringPre = ["Mail", "E-Mail", "e-Mail"]

def contactBlock():
    name = fake.company()
    address = fake.address()
    person = fake.name()
    telInt = random.randint(1,8)
    telS = fake.phone_number() 
    tel = telS + str(telInt)
    fax = telS + str(telInt+1)
    cl = random.choice(col)

    regName = regex.sub('', name.lower())
    domain = regName + random.choice(webEnd)
    mail = random.choice(mailPre) + domain
    domain = random.choice(webPre) + domain

    if random.randint(0,1) == 0:
        person = random.choice(personPre) + cl + person
        tel = random.choice(telPre) + cl + tel
        fax = "Fax" + cl + fax
        mail = random.choice(mailStringPre) + cl + mail
        domain = random.choice(webStringPre) + cl + domain

    block = name + "\n" + address + "\n" + tel + "\n" + fax + "\n" + person + "\n" + mail + "\n" + domain
    return block

def nameBot():
    cl = random.choice(col)
    name = fake.company()
    address = fake.address()
    person = "Inh." + cl + fake.name()

    block = name + "\n" + address + "\n" + person
    return block

def contactBot():
    name = fake.company()
    person = fake.name()
    telInt = random.randint(1,8)
    telS = fake.phone_number() 
    tel = telS + str(telInt)
    fax = telS + str(telInt+1)
    cl = random.choice(col)

    regName = regex.sub('', name.lower())
    domain = regName + random.choice(webEnd)
    mail = random.choice(mailPre) + domain
    domain = random.choice(webPre) + domain

    if random.randint(0,1) == 0:
        person = random.choice(personPre) + cl + person
        tel = random.choice(telPre) + cl + tel
        fax = "Fax" + cl + fax
        mail = random.choice(mailStringPre) + cl + mail
        domain = random.choice(webStringPre) + cl + domain

    block = person + "\n" + tel + "\n" + fax + "\n" + mail + "\n" + domain
    return block

def topText():
    preText = "Sehr geehrter Herr " + fake.last_name_male() + ", \n"
    preText = preText + "vielen Dank für Ihren Auftrag und das damit verbundene Vertrauen! \n"
    text = random.choice(topTextSyn)
    if random.randint(0,1) == 0:
        text = preText + text
    return text

def botText():
    return random.choice(botTextSyn)

def bank():
    cl = random.choice(col)
    preText = "Bankverbindung" + cl 
    bank = random.choice(bankSyn)
    iban = "IBAN" + cl + fake.iban()
    bic = "BIC" + cl + fake.swift()
    block = preText +"\n" + iban + "\n" + bic
    return block

def taxOffice():
    cl = random.choice(col)
    preText = "Finanzamt" + cl 
    finanzamt = "Finanzamt" + cl + fake.city()
    taxNr ="Steuernummer" + cl + str(random.randint(100,999)) + "/" + str(random.randint(1000,9999)) + "/" + str(random.randint(100,999))
    block = preText + "\n" + finanzamt + "\n" + taxNr
    return block

def KDN():
    kdn = random.choice(kdnSyn) + random.choice(col) + str(random.randint(1000,999999))
    return kdn

# cpN = [fake.unique.company() for i in range(100000)]
# print(len(cpN))
# print(len(set(cpN)))