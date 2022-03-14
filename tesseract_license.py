'''
To improve the quality/recall rate, change the engine:
export TESSDATA_PREFIX=/home/leo/git/tesseract-ocr-eng/tessdata_best/
'''

import pytesseract
import imutils
from PIL import Image


img = Image.open('./ocr_test4_crop.jpg')

# Resize to improve the recall rate
# height of capitcal letter 20~100 px
# basewidth = 100
# wpercent = (basewidth/float(img.size[0]))
# hsize = int((float(img.size[1])*float(wpercent)))
# img = img.resize((basewidth,hsize), Image.ANTIALIAS)

config = ("-l eng --psm 7 --dpi 200")
text = pytesseract.image_to_string(img, config=config)

print(text)
print(pytesseract.image_to_data(img, config=config))
print(pytesseract.image_to_osd(img, config='--psm 0 -c min_characters_to_try=3'))

img.show()