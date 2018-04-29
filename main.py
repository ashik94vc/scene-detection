import os
import re
import sys
import warnings
import numpy as np

from PIL import Image
from math import floor
from scipy.misc import toimage

from core.lib import loadModel,loadData
from core.classifier import Classifier
from core.model import Model
from core.lib import saveModel

dataset = loadData('dataset/imagenet.h5')
# zeros_array = np.zeros(200,dtype=int)
# for i in range(len(dataset['train']['target'])):
#     temp = zeros_array
#     temp[199-dataset['train']['target'][i]] = 1
#     dataset['train']['target'][i] = temp
classifier = Classifier(dataset)
parameters = None

if len(sys.argv) < 2:
    classifier.train()
else:
    parameters = loadModel(sys.argv[1])
    # classifier.model = Model(parameters)
if len(sys.argv) > 2:
    predict_input = (np.asarray(Image.open(open(sys.argv[2], 'rb')),dtype="float64")/256).transpose(2,0,1)
    y_output = Model(parameters).predict([predict_input])
    # y_output_scalar = np.asscalar(y_output)
    out = np.argmax(y_output)
    print(out)
    a = y_output.argsort()[0][-5:][::-1]
    print(dataset["class_index"][out])
    print("Other Classes")
    print(a)
    for i in a:
        print(dataset["class_index"][i])
    # string_output = "I am {:.3f}% sure that it's a ".format(confidence(y_output_scalar))
    # print(string_o/utput+num_to_label(y_output_scalar))
else:
    error = classifier.test()
    print(error)
