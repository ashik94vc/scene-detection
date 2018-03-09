from classifier import Classifier
from PIL import Image
from scipy.misc import toimage
import os
import numpy as np
import re
from math import floor

#build data for cars
regex = re.compile("^cars_[0-9]{5}\.jpg$")
dataset_temp = list()
for filename in os.listdir('../dataset/cars'):
    if regex.match(filename):
        file_data = Image.open(open(os.path.join('../dataset/cars',filename),'rb'))
        img = np.asarray(file_data, dtype='float64')/256
        img = img.transpose(2,0,1)
        dataset_temp.append([img,1])
#build data for not cars_
regex = re.compile("^notcars_[0-9]{5}\.jpg$")
nocars_list = list()
for filename in os.listdir('../dataset/not_cars'):
    if regex.match(filename):
        file_data = Image.open(open(os.path.join('../dataset/not_cars',filename),'rb'))
        img = np.asarray(file_data, dtype='float64')/256
        img = img.transpose(2,0,1)
        dataset_temp.append([img,0])
dataset_temp = np.asarray(dataset_temp)
np.random.shuffle(dataset_temp)
dataset = []
size_d = len(dataset_temp)
train_size = size_d*0.7
test_size = 0.3*size_d
train_index = floor(train_size)
test_index = size_d - train_index
train_set = dataset_temp[0:train_index]
test_set = dataset_temp[train_index+1:test_index]
train_inputs,train_labels = np.asarray([train[0] for train in train_set]),np.asarray([train[1] for train in train_set])
test_inputs,test_labels = np.asarray([test[0] for test in test_set]),np.asarray([test[1] for test in test_set])
dataset.append((train_inputs,train_labels))
dataset.append((test_inputs,test_labels))
classifier = Classifier(dataset)
classifier.train()
error = classifier.test()
