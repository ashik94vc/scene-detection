from model import train as conv_nn_train
from model import predict as conv_nn_predict
import theano
from theano import tensor as T
from model import model
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

    def error(self, Y_actual, Y_output):
        Y = Y_actual - Y_output
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
            cost,parameters,grads = conv_nn(self.train_x[i].reshape(1,3,150,150),y_label=self.train_y[i],parameters=parameters)
            # grads = self.compute_gradients(np.asscalar(cost),parameters)
            iterations = 0
            while np.linalg.norm(grads["dW1"])**2 > self.eps:
                parameters["W1"] = parameters["W1"] - self.learning_rate*grads["dW1"]
                print(updated)
                iterations += 1
                if iterations >= 1000:
                    break
            iterations = 0
            while np.linalg.norm(grads["db1"])**2 > self.eps:
                parameters["b1"] = parameters["b1"] - self.learning_rate*grads["db1"]
                iterations += 1
                if iterations >= 1000:
                    break
            iterations = 0
            while np.linalg.norm(grads["dW2"])**2 > self.eps:
                parameters["W2"] = parameters["W2"] - self.learning_rate*grads["dW2"]
                iterations += 1
                if iterations >= 1000:
                    break
            iterations = 0
            while np.linalg.norm(grads["db2"])**2 > self.eps:
                parameters["b2"] = parameters["b2"] - self.learning_rate*grads["db2"]
                iterations += 1
                if iterations >= 1000:
                    break
            iterations = 0
            while np.linalg.norm(grads["dW3"])**2 > self.eps:
                parameters["W3"] = parameters["W3"] - self.learning_rate*grads["dW3"]
                iterations += 1
                if iterations >= 1000:
                    break
            iterations = 0
            while np.linalg.norm(grads["db3"])**2 > self.eps:
                parameters["b3"] = parameters["b3"] - self.learning_rate*grads["db3"]
                iterations += 1
                if iterations >= 1000:
                    break
            iterations = 0
            while np.linalg.norm(grads["dW5"])**2 > self.eps:
                parameters["W5"] = parameters["W5"] - self.learning_rate*grads["dW5"]
                iterations += 1
                if iterations >= 1000:
                    break
            print(iterations)
            iterations = 0
            while np.linalg.norm(grads["dW6"])**2 > self.eps:
                parameters["W6"] = parameters["W6"] - self.learning_rate*grads["dW6"]
                iterations += 1
                if iterations >= 1000:
                    break
            print(i)
        self.params = parameters

    def test(self):
        Y_predict = list()
        for x in self.test_x:
            y_predict = conv_nn(x,self.params)
            Y_predict.append(y_predict)
        Y_predict = np.matrix(Y_predict)
        error = self.error(self.test_y,Y_predict)
        return error

    def cost_function(self, y_predict, y_label):
        return y_label*np.log(y_predict) + (1-y_label)*np.log(1-y_predict)
