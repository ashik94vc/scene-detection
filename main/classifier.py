import model
import theano
from theano import tensor as T
from model import model
from scipy.misc import toimage

def error(Y_actual, Y_output):
    Y = Y_actual - Y_output
    error = (1/2)*(T.transpose(Y)*Y)
    return error

def negative_log_likelihood(Y):
    return -T.mean(T.log(Y)[T.arange(Y.shape[0]), Y])
