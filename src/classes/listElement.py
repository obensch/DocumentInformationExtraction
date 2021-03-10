class ListElement:
    id = ""
    content = ""
    pdfX = 0
    pdfY = 0
    itemNumber = None
    startX = None
    startY = None
    endX = None
    endY = None

    def __init__(self, Id):
        self.id = Id