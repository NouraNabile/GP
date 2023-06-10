#!/usr/bin/env python
# coding: utf-8

# In[1]:
#import os
#import cv2
import imutils
import numpy as np 
import pandas as pd
import cv2
import os


from PIL import Image,ImageEnhance
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.models import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import matplotlib.pyplot as plt
from tensorflow import keras
from tqdm import tqdm
import random

from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,CSVLogger



# In[3]:
def getPrediction(filename):

    def open_images(paths):

        images = []
        for path in paths:
            image = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            image = augment_image(image)
            images.append(image)
        return np.array(images)

    def crop_img(img):
       # """
        # Finds the extreme points on the image and crops the rectangular out of them
        # """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold the image, then perform a series of erosions + dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

    # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        ADD_PIXELS = 0
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

        return new_img

    def augment_image(image):
        image = Image.fromarray(np.uint8(image))
        image = ImageEnhance.Brightness(
            image).enhance(random.uniform(0.8, 1.2))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.8, 1.2))
        image = np.array(image)/255.0
        return image

    IMAGE_SIZE = 224
# test_dir = 'neworiginaldata/testing'

# unique_labels = os.listdir(test_dir)
    unique_labels = ['butterfly glioblastoma',
                     'meningioma', 'notumor', 'pituitary']

    def datagen(paths, labels, batch_size=12, epochs=1):

        for _ in range(epochs):
            for x in range(0, len(paths), batch_size):
                batch_paths = paths[x:x+batch_size]
                batch_images = open_images(batch_paths)
                batch_labels = labels[x:x+batch_size]
                batch_labels = encode_label(batch_labels)
                yield batch_images, batch_labels

    def encode_label(labels):
        encoded = []
        for x in labels:
            encoded.append(unique_labels.index(x))
        return np.array(encoded)

    def decode_label(labels):
        decoded = []
        for x in labels:
            decoded.append(unique_labels[x])
        return np.array(decoded)

# inception 99.3%
   # model1 = load_model('model/newmodelv3usingaugmenteddataset50epoch99acc.h5',
   #                     custom_objects={'BatchNormalization': BatchNormalization})
   

   # custom_objects = {
    #     'BatchNormalization': BatchNormalization}
    #model1 = load_model(
     #    'model/newmodelv3usingaugmenteddataset50epoch99acc.h5', custom_objects=custom_objects)
    model1 = load_model('model/newmodelv3usingaugmenteddataset50epoch99acc.h5')
# xception 99.2%
   # model2 = load_model('model/newmodelXceptionusingaugmenteddataset50epoch99acc.h5')

# vgg16 98.4%
    # model3 = load_model('model/newvgg16modelusingaugdatadataset50epoch984accuracy.h5')
# vgg19 98.1%
    # model4 = load_model('model/newmodelvgg19usingaugmentationset50epoch981acc.h5')
    path = 'static/brainimages/'+filename
    IMAGE_SIZE = 224

    # image = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.imread(path)
    image = crop_img(image)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
   # image = crop_img(image)
    image = augment_image(image)
    img = np.array(image)

    img = np.expand_dims(img, axis=0)
    # print(img)

    result1 = model1.predict(img)
    # result2=model2.predict(img)
   # result3=model3.predict(img)
   # result4=model4.predict(img)

    predss = [result1]  # ,result2,result3,result4]
    # print(np.shape(predss))
    summed = np.sum(predss, axis=0)

    # argmax across classes
    ensemble_prediction = np.argmax(summed, axis=1)

    pred = decode_label(ensemble_prediction)
   # print(pred)
    return pred


# In[13]:
test_prediction = getPrediction('images_1.jpg')
print(test_prediction)


# In[5]:


# filename="modelaftervalone.h5 "
# loaded_model=load_model(filename)
# image=cv2.imread("braintumor/Training/glioma/Tr-glTr_0000.jpg")
# img=augment_image(image)
# img=img.resize((224,224))
# img=np.array(img)


# In[6]:


# In[7]:


# In[8]:


# In[9]:


# In[10]:


# In[11]:


# In[14]:


# # weight prediction

# In[75]:


# weights = [0.5,0.4, 0.3, 0.1]

# Use tensordot to sum the products of all elements over specified axes.
# weighted_preds = np.tensordot(predss, weights, axes=((0),(0)))
# weighted_ensemble_prediction = np.argmax(weighted_preds, axis=1)


# In[76]:


# pred1=decode_label(weighted_ensemble_prediction)
# print(pred1)


# In[ ]:


# print(ensemble_prediction)


# In[ ]:


# print(summed)


# In[ ]:


# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# print('\nAccuracy:  {:.3f}  \n'.format(summed))


# In[ ]:


# pred = np.argmax(result, axis=-1)
# pred=decode_label(pred)
# print(pred)
