import os
from core.model import Model
import theano
from theano import tensor as T
from scipy.misc import toimage
from scipy.optimize import fmin_cg as conjugate_gradient
from core.lib import saveModel
import numpy as np
import pickle

class Classifier(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.train_x,self.train_y = dataset[0]
        self.test_x,self.test_y = dataset[1]
        self.learning_rate = 0.16
        self.eps = 2e-9
        self.params = {}
        self.model = Model()

    def error(self, Y_actual, Y_output):
        Y = Y_actual - Y_output
        error = (1/Y.shape[0])*np.sum(Y*Y.T)
        print(error.shape)
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
        for i in range(len(self.train_x)):
            cost = self.model.train([self.train_x[i]],[self.train_y[i]])
            if i%100 == 0:
                print(str(i)+" datas trained")
        self.params = parameters
        saveModel(parameters)


    def test(self, parameters = None):
        Y_predict = list()
        if parameters != None:
            self.params = parameters
        for x in self.test_x[:100]:
            y_predict = self.model.predict([x])
            Y_predict.append(np.asscalar(y_predict))
        Y_predict = np.asarray(Y_predict)
        error = self.error(self.test_y[:100],Y_predict)
        return error

    def cost_function(self, y_predict, y_label):
        return y_label*np.log(y_predict) + (1-y_label)*np.log(1-y_predict)