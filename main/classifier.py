from model import model as conv_nn
import theano
from theano import tensor as T
from model import model
from scipy.misc import toimage
from scipy.optimize import fmin_cg as conjugate_gradient

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

    def compute_gradients(cost, parameters):
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

    def train():
        train_y = self.train_y.iter()
        parameters = None
        for x in self.train_x:
            y_predict,parameters = conv_nn(x,parameters)
            cost = -cost_function(y_predict,train_y.next())
            grads = compute_gradients(cost,grads)
            iterations = 0
            while np.linalg.norm(grads["W1"])**2 > eps:
                parameters["W1"] = parameters["W1"] - self.learning_rate*grads["W1"]
                iterations += 1
                if iterations >= 1000:
                    break
            iterations = 0
            while np.linalg.norm(grads["b1"])**2 > eps:
                parameters["b1"] = parameters["b1"] - self.learning_rate*grads["b1"]
                iterations += 1
                if iterations >= 1000:
                    break
            iterations = 0
            while np.linalg.norm(grads["W2"])**2 > eps:
                parameters["W2"] = parameters["W2"] - self.learning_rate*grads["W2"]
                iterations += 1
                if iterations >= 1000:
                    break
            iterations = 0
            while np.linalg.norm(grads["b2"])**2 > eps:
                parameters["b2"] = parameters["b2"] - self.learning_rate*grads["b2"]
                iterations += 1
                if iterations >= 1000:
                    break
            iterations = 0
            while np.linalg.norm(grads["W3"])**2 > eps:
                parameters["W3"] = parameters["W3"] - self.learning_rate*grads["W3"]
                iterations += 1
                if iterations >= 1000:
                    break
            iterations = 0
            while np.linalg.norm(grads["b3"])**2 > eps:
                parameters["b3"] = parameters["b3"] - self.learning_rate*grads["b3"]
                iterations += 1
                if iterations >= 1000:
                    break
            iterations = 0
            while np.linalg.norm(grads["W5"])**2 > eps:
                parameters["W5"] = parameters["W5"] - self.learning_rate*grads["W5"]
                iterations += 1
                if iterations >= 1000:
                    break
            iterations = 0
            while np.linalg.norm(grads["W6"])**2 > eps:
                parameters["W6"] = parameters["W6"] - self.learning_rate*grads["W6"]
                iterations += 1
                if iterations >= 1000:
                    break
        return parameters

    def test(parameters):
        Y_predict = list()
        for x in self.test_x:
            y_predict,parameters = conv_nn(x,parameters)
            Y_predict.append(y_predict)
        Y_predict = np.matrix(Y_predict)
        error = self.error(self.test_y,Y_predict)
        return error

    def cost_function(y_predict, y_label):
        return y_label*np.log(y_predict) + (1-y_label)*np.log(1-a)
