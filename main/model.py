import theano
from theano import tensor
import numpy as np
from theano.tensor.nnet import conv2d,relu,sigmoid
from theano.tensor.signal.pool import pool_2d
from PIL import Image
from scipy.misc import toimage

def RMSprop(cost, parameters, learning_rate=0.011, rho=0.8, epsilon=1e-6):
    gradients = tensor.grad(cost, parameters)
    updates = []
    for param, grad in zip(parameters, gradients):
        acc = theano.shared(param.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * grad ** 2
        gradient_scaling = tensor.sqrt(acc_new + epsilon)
        grad = grad / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((param, param - learning_rate * grad))
    return updates

__X = tensor.tensor4()
Y = tensor.matrix()
__W1 = theano.shared(np.random.randn(6,3,9,9))
__b1 = theano.shared(np.zeros(6,).astype(theano.config.floatX))
__W2 = theano.shared(np.random.randn(32,6,9,9))
__b2 = theano.shared(np.zeros(32,).astype(theano.config.floatX))
__W3 = theano.shared(np.random.randn(64,32,9,9))
__b3 = theano.shared(np.zeros(64,).astype(theano.config.floatX))
__W5 = theano.shared(np.random.randn(7744,84))
# __b5 = theano.shared(np.zeros(64*9*9,))
__W6 = theano.shared(np.random.randn(84,1))
# __b6 = theano.shared(np.zeros(84,))
# if parameters == None:
#     parameters = {}
#
#     parameters["W1"] = __W1
#     parameters["b1"] = __b1
#     parameters["W2"] = __W2
#     parameters["b2"] = __b2
#     parameters["W3"] = __W3
#     parameters["b3"] = __b3
#     parameters["W5"] = __W5
#     parameters["W6"] = __W6
# else:
#     __W1 = parameters["W1"]
#     __b1 = parameters["b1"]
#     __W2 = parameters["W2"]
#     __b2 = parameters["b2"]
#     __W3 = parameters["W3"]
#     __b3 = parameters["b3"]
#     __W5 = parameters["W5"]
#     __W6 = parameters["W6"]

__layer_1 = conv2d(__X,__W1)
__layer_1_pool = pool_2d(__layer_1,(2,2),ignore_border=True)
__layer_1_output = relu(__layer_1_pool+__b1.dimshuffle('x', 0, 'x', 'x'))

__layer_2 = conv2d(__layer_1_output, __W2)
__layer_2_pool = pool_2d(__layer_2,(2,2),ignore_border=True)
__layer_2_output = relu(__layer_2_pool+__b2.dimshuffle('x', 0, 'x', 'x'))

__layer_3 = conv2d(__layer_2_output, __W3)
__layer_3_pool = pool_2d(__layer_3,(2,2),ignore_border=True)
__layer_3_output = relu(__layer_3_pool+__b3.dimshuffle('x', 0, 'x', 'x'))

__layer_4 = __layer_3_output.flatten(2)

__layer_5 = tensor.dot(__layer_4,__W5)
__layer_5_output = __layer_5.tanh()

__layer_6 = tensor.dot(__layer_5_output, __W6)
__layer_6_output = sigmoid(__layer_6)

    # if y_label != None:
    #     grads = {}
    #     cost = -(y_label*np.log(__layer_6_output) + (1-y_label)*np.log(1-__layer_6_output)).sum()
    #     grads["dW1"] = tensor.grad(cost,__W1).eval()
    #     grads["db1"] = tensor.grad(cost,__b1).eval()
    #     grads["dW2"] = tensor.grad(cost,__W2).eval()
    #     grads["db2"] = tensor.grad(cost,__b2).eval()
    #     grads["dW3"] = tensor.grad(cost,__W3).eval()
    #     grads["db3"] = tensor.grad(cost,__b3).eval()
    #     grads["dW5"] = tensor.grad(cost,__W5).eval()
    #     grads["dW6"] = tensor.grad(cost,__W6).eval()
    #     return cost.eval(),parameters,grads
    # return __layer_6_output.eval(),parameters
cost = -(Y*np.log(__layer_6_output) + (1-Y)*np.log(1-__layer_6_output)).sum()

parameters = [__W1,__b1,__W2,__b2,__W3,__b3,__W5,__W6]

updates = RMSprop(cost,parameters)

train = theano.function([__X, Y], cost,updates=updates)
predict = theano.function([__X],__layer_6_output)
