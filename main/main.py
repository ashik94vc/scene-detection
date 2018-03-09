from classifier import Classifier
from PIL import Image
from scipy.misc import toimage
import os
import numpy as np
import re

#build data for cars
regex = re.compile("^cars_[0-9]{5}\.jpg$")
cars_list = list()
for filename in os.listdir('../dataset/cars'):
    if regex.match(filename):
        file_data = Image.open(open(os.path.join('../dataset/cars',filename),'rb'))
        img = np.asarray(file_data, dtype='float64')/256
        img = img.transpose(2,0,1)
        cars_list.append([img,1])
cars_data = np.asarray(cars_list)
print(cars_data.shape)
# classifier = Classifier()
