import os
import re
import sys
import warnings
import numpy as np

from PIL import Image
from math import floor
from scipy.misc import toimage

from dataset.imagenet_processor import ImageNet
from core.lib import loadModel
from core.classifier import Classifier
from core.model import Model

# def quantizer(value):
#     if value > 0.5:
#         return 1
#     return 0
#
# def confidence(value):
#     if value > 0.5:
#         return value*100
#     return (1.0-value)*100
#
# def num_to_label(input_value):
#     if input_value > 0.5:
#         return "a car!"
#     else:
#         return "not a car!"
#
# #build data for cars
# regex = re.compile("^cars_[0-9]{5}\.jpg$")
# dataset_temp = list()
# for filename in os.listdir('dataset/cars'):
#     if regex.match(filename):
#         file_data = Image.open(open(os.path.join('dataset/cars',filename),'rb'))
#         img = np.asarray(file_data, dtype='float64')/256
#         img = img.transpose(2,0,1)
#         dataset_temp.append([img,1])
# #build data for not cars_
# regex = re.compile("^notcars_[0-9]{5}\.jpg$")
# nocars_list = list()
# for filename in os.listdir('dataset/not_cars'):
#     if regex.match(filename):
#         file_data = Image.open(open(os.path.join('dataset/not_cars',filename),'rb'))
#         img = np.asarray(file_data, dtype='float64')/256
#         img = img.transpose(2,0,1)
#         dataset_temp.append([img,0])
# dataset_temp = np.asarray(dataset_temp)
# np.random.shuffle(dataset_temp)
# dataset = []
# size_d = len(dataset_temp)
# train_size = size_d*0.7
# test_size = 0.3*size_d
# train_index = floor(train_size)
# train_set = dataset_temp[0:train_index]
# test_set = dataset_temp[train_index+1:size_d]
# train_inputs,train_labels = np.asarray([train[0] for train in train_set]),np.asarray([train[1] for train in train_set])
# test_inputs,test_labels = np.asarray([test[0] for test in test_set]),np.asarray([test[1] for test in test_set])
# dataset.append((train_inputs,train_labels))
# dataset.append((test_inputs,test_labels))

dataset = []
imageNet = ImageNet()

train_inputs, train_labels = imageNet.train_inputs, imageNet.train_labels
test_inputs, test_labels = imageNet.test_inputs, imageNet.test_labels

dataset.append((train_inputs, train_labels))
dataset.append((test_inputs, test_labels))

print(train_inputs.shape, train_labels.shape)
classifier = Classifier(dataset)
parameters = None

if len(sys.argv) < 2:
    classifier.train()
else:
    parameters = loadModel(sys.argv[1])
    classifier.model = Model(parameters)
if len(sys.argv) > 2:
    predict_input = (np.asarray(Image.open(open(sys.argv[2], 'rb')),dtype="float64")/256).transpose(2,0,1)
    y_output = Model(parameters).predict([predict_input])
    # y_output_scalar = np.asscalar(y_output)
    print(y_output)
    # string_output = "I am {:.3f}% sure that it's a ".format(confidence(y_output_scalar))
    # print(string_o/utput+num_to_label(y_output_scalar))
else:
    error = classifier.test()
    print(error)
