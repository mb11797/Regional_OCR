import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50

model = resnet50.ResNet50()

img = image.load_img("image.jpg", target_size = (224,224))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = resnet50.preprocess_input(x)

predictions = model.predict(x)

predicted_classes = resnet50.decode_predictions(predictions, top=9)

print("This is an image of : ")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood))











