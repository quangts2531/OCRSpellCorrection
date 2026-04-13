import pymupdf
import re


doc = pymupdf.open("cv.pdf") # open a document

full_text = ""
for page in doc: # iterate the document pages
  full_text += page.get_text() #

full_text = re.sub(r'\n(?![A-Z])', ' ', full_text)
print(full_text)