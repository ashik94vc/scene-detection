import model
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
        self.batch_size = 1

    def error(self, Y_actual, Y_output):
        Y = Y_actual - Y_output
        error = (1/2)*(T.transpose(Y)*Y)
        return error

    def train():
        for x in train_x:
            

    def negative_log_likelihood(Y):
        return -T.mean(T.log(Y)[T.arange(Y.shape[0]), Y])
