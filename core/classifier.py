import os
from core.model import Model
import theano
from theano import tensor as T
from scipy import ndimage
from scipy.misc import toimage
from scipy.optimize import fmin_cg as conjugate_gradient
from keras.preprocessing.image import ImageDataGenerator
from core.lib import saveModel
import numpy as np
import pickle

class Classifier(object):

    def __init__(self, dataset):
        self.imagegen = ImageDataGenerator(shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
        self.dataset = dataset
        self.train_x,self.train_y = dataset[0]
        self.test_x,self.test_y = dataset[1]
        self.learning_rate = 0.16
        self.eps = 2e-9
        self.params = {}
        self.model = Model()

    def error(self, Y_actual, Y_output):
        Y = Y_actual - Y_output
        print(Y)
        error = (1/Y.shape[0])*np.sum(Y*Y.T)
        return error

    def compute_gradients(self, cost, parameters):
        grads = {}
        grads["dW1"] = T.grad(cost,parameters["W1"])
        grads["db1"] = T.grad(cost,parameters["b1"])
        grads["dW2"] = T.grad(cost,parameters["W2"])
        grads["db2"] = T.grad(cost,parameters["b2"])
        grads["dW3"] = T.grad(cost,parameters["W3"])
        grads["db3"] = T.grad(cost,parameters["b3"])
        grads["dW5"] = T.grad(cost,parameters["W5"])
        grads["dW6"] = T.grad(cost,parameters["W6"])

        return grads

    def train(self):
        parameters = None
        num_epoch = 1
        while num_epoch > 0:
            mini_batch_iter = self.imagegen.flow(self.train_x,self.train_y,batch_size=16)
            current_data = 1
            for train_batch in mini_batch_iter:
                # sh = self.model.shape_dim(train_batch[0],train_batch[1])
                cost = self.model.train(train_batch[0],train_batch[1])
                if current_data%100 == 0:
                    print(str(current_data*16)+" datas trained")
                if current_data == len(self.train_x):
                    break
                current_data += 1
            error = self.model.test(self.test_x.astype(theano.config.floatX),self.test_y)
            print("Epoch "+str(2-num_epoch)+" done")
            print("Error: "+str(error))
            num_epoch -= 1
        self.params = self.model.parameters()
        saveModel(self.params)

    def test(self, parameters = None):
        error = self.model.test(self.test_x.astype(theano.config.floatX), self.test_y)
        return error
        # error = self.error(self.test_y,Y_predict)
        # return error

    def cost_function(self, y_predict, y_label):
        return y_label*np.log(y_predict) + (1-y_label)*np.log(1-y_predict)
