from PIL import Image
import pytesseract

#Print the text detected and recognised
print(pytesseract.image_to_string(Image.open("image_hin+eng1.jpg"), lang='hin+eng'))
# print(type(pytesseract.image_to_string(Image.open("image_hin+eng1.jpg"), lang='hin+eng')))


print(pytesseract.image_to_boxes(Image.open("image_hin+eng1.jpg"), lang='hin+eng'))

# print(type(pytesseract.image_to_boxes(Image.open("image_hin+eng1.jpg"), lang='hin+eng')))
bbox = pytesseract.image_to_boxes(Image.open("image_hin+eng1.jpg"), lang='hin+eng')
# print(type(bbox))
print("\n\n\nLength of the string bbox : ", len(bbox))
# print("\n\n\n of the string bbox : ", size(bbox))
# bbox = list(bbox)
# print(type(bbox))
# print(bbox)

print('\n\ntesseract version : ', pytesseract.get_tesseract_version())

# print(pytesseract.image_to_data(Image.open("image_hin+eng1.jpg"), 'hin+eng'))

# print(pytesseract.image_to_osd(Image.open("image.jpg")))


