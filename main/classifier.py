from model import train as conv_nn_train
from model import predict as conv_nn_predict
import theano
from theano import tensor as T
from scipy.misc import toimage
from scipy.optimize import fmin_cg as conjugate_gradient
import numpy as np

class Classifier(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.train_x,self.train_y = dataset[0]
        self.test_x,self.test_y = dataset[1]
        self.learning_rate = 0.16
        self.eps = 2e-9
        self.params = {}

    def error(self, Y_actual, Y_output):
        Y = Y_actual - Y_output
        print(Y.shape)
        error = (1/2)*(T.transpose(Y)*Y)
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
            cost = conv_nn_train([self.train_x[i]],[self.train_y[i]])
            if i%100 == 0:
                print(str(i)+" datas trained")
        self.params = parameters
        np.save('train_file.model',parameters)

    def test(self, parameters = None):
        Y_predict = list()
        if parameters != None:
            self.params = parameters
        for x in self.test_x:
            y_predict = conv_nn_predict([x],self.params)
            print('hello')
            print(y_predict)
            Y_predict.append(y_predict)
        Y_predict = np.asarray(Y_predict)
        error = self.error(self.test_y,Y_predict)
        return error

    def cost_function(self, y_predict, y_label):
        return y_label*np.log(y_predict) + (1-y_label)*np.log(1-y_predict)
