from PIL import Image
import pytesseract

#Print the text detected and recognised
print(pytesseract.image_to_string(Image.open("image_cropped.jpg")))


# print(pytesseract.image_to_boxes(Image.open("image1.png")))

# print(pytesseract.image_to_data(Image.open("image.jpg")))

# print(pytesseract.image_to_osd(Image.open("image.jpg")))

