class HeaderElement:
    
    id = ""
    content = ""
    fontName = ""
    fontSize = ""
    fontColor = ""
    pdfX = 0
    pdfY = 0
    startX = None
    startY = None
    endX = None
    endY = None

    def __init__(self, Id):
        self.id = Id