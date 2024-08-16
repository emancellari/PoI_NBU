## Here we extract the noise from adversarial image (0.55-strong) then upscale the noise to the size of the original image
## and add on the original image. Then check if it is still adversarial or not. By this, we will have and HR adversarial image.
import glob
import os

import PIL
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
import cv2
from scipy import interpolate
from keras_preprocessing.image import load_img, img_to_array

from keras.applications.mobilenet_v2 import (
    decode_predictions,
    preprocess_input,
    MobileNetV2,
)

model1 = MobileNetV2(weights='imagenet')

adversarial=["test_adversarial_image/acorn1_adv_top35_eps16.png"]
original=["test_clean_image/acorn1.JPEG"]

for i in range (len(adversarial)):
    # def noiseEnlargement(orgLarge, advSmall, interp, dest):
    advSmall = load_img(f"{adversarial[i]}",target_size=(224,224), interpolation='lanczos')

    original_image=load_img(f"{original[i]}")
    x,y = original_image.size

    print(x,y)
    imgAdvSmall = img_to_array(advSmall)
    orgLarge_down = load_img(f"{original[i]}",target_size=(224,224), interpolation='lanczos')



    orgLarge_down = img_to_array(orgLarge_down)

    # L.L.L resize in H domain  upsize


    org_H = cv2.resize(orgLarge_down,(x,y), interpolation=cv2.INTER_LANCZOS4)

    orgLarge_im = np.clip(org_H, 0, 255)
    orgLarge_im = Image.fromarray((orgLarge_im).astype(np.uint8))


    #What is the ct of clean image
    original_testing = load_img(f"{original[i]}", target_size=(224, 224), interpolation='lanczos')

    original_testing = img_to_array(original_testing)
    image = original_testing.reshape((1, 224, 224, 3))
    yhat = model1.predict(preprocess_input(image))
    label0 = decode_predictions(yhat)
    label1 = label0[0][0]
    print("a) -> original_testing: %s  %.4f%%" % (label1[1], label1[2]))
    print(label0)
    print(label1[1], label1[2])


    #take original image and apply noise blowing up method
    w, h = load_img(f"{original[i]}").size
    orgLarge=load_img(f"{original[i]}")




    orgSmall_im = orgLarge.resize((224,224), Image.LANCZOS)

    orgSmall = img_to_array(orgSmall_im)

    #exctract the noise

    noiseSmall = advSmall - orgSmall
    noiseLarge = cv2.resize(noiseSmall, (w,h), interpolation=cv2.INTER_LANCZOS4)





    advLarge = orgLarge + noiseLarge
    advLarge = np.clip(advLarge, 0, 255)
    advLarge_im = Image.fromarray((advLarge).astype(np.uint8))
    advLarge_im.save(f"{original[i]}_noise_adversarial.png", 'png')

    imgAdvLarge_imSaved = load_img(f"{original[i]}_noise_adversarial.png", target_size=(224,224), interpolation='lanczos')

    imgAdvLarge_imSaved= img_to_array(imgAdvLarge_imSaved)
    image = imgAdvLarge_imSaved.reshape((1, 224, 224, 3))
    yhat = model1.predict(preprocess_input(image))
    label0 = decode_predictions(yhat)
    label1 = label0[0][0]
    print("d) -> The CT after: %s  %.4f%%" % (label1[1], label1[2]))
    print(label0)
    print(label1[1], label1[2])



    a=label1[2]





    imgAdvSmall = img_to_array(advSmall)
    image = imgAdvSmall.reshape((1, 224, 224, 3))
    yhat = model1.predict(preprocess_input(image))
    label0 = decode_predictions(yhat)
    label1 = label0[0][0]
    print("The CT before: %s  %.4f%%" % (label1[1], label1[2]))
    print(label0)
    print(label1[1], label1[2])
    b=label1[2]

    loss= a-b

    print(loss)