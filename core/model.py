import theano
from theano import tensor
import numpy as np
from theano.tensor.nnet import conv2d,relu,sigmoid,softmax
from theano.tensor.signal.pool import pool_2d
from PIL import Image
from scipy.misc import toimage

class Model(object):
    def __init__(self,parameters=None):
        X = tensor.tensor4()
        Y = tensor.lvector()
        self.params = parameters
        if parameters == None:
            W1 = theano.shared(np.random.randn(32,3,5,5))
            b1 = theano.shared(np.zeros(32,).astype(theano.config.floatX))
            W2 = theano.shared(np.random.randn(64,32,5,5))
            b2 = theano.shared(np.zeros(64,).astype(theano.config.floatX))
            W3 = theano.shared(np.random.randn(128,64,5,5))
            b3 = theano.shared(np.zeros(128,).astype(theano.config.floatX))
            W5 = theano.shared(np.random.randn(28800,1084))
            # b5 = theano.shared(np.zeros(64*9*9,))
            W6 = theano.shared(np.random.randn(1084,2))
            b6 = theano.shared(np.zeros(2,))
        else:
            W1 = theano.shared(parameters["W1"])
            b1 = theano.shared(parameters["b1"])
            W2 = theano.shared(parameters["W2"])
            b2 = theano.shared(parameters["b2"])
            W3 = theano.shared(parameters["W3"])
            b3 = theano.shared(parameters["b3"])
            W5 = theano.shared(parameters["W5"])
            W6 = theano.shared(parameters["W6"])
            b6 = theano.shared(parameters["b6"])

        layer_1 = conv2d(X,W1)
        layer_1_pool = pool_2d(layer_1,(2,2),ignore_border=True)
        layer_1_output = tensor.tanh(layer_1_pool+b1.dimshuffle('x', 0, 'x', 'x'))

        layer_2 = conv2d(layer_1_output, W2)
        layer_2_pool = pool_2d(layer_2,(2,2),ignore_border=True)
        layer_2_output = tensor.tanh(layer_2_pool+b2.dimshuffle('x', 0, 'x', 'x'))

        layer_3 = conv2d(layer_2_output, W3)
        layer_3_pool = pool_2d(layer_3,(2,2),ignore_border=True)
        layer_3_output = tensor.tanh(layer_3_pool+b3.dimshuffle('x', 0, 'x', 'x'))

        layer_4 = layer_3_output.flatten(2)

        layer_5 = tensor.dot(layer_4,W5)
        layer_5_output = layer_5.tanh()

        layer_6 = tensor.dot(layer_5_output, W6) + b6

        #softmax instead of sigmoid.
        layer_6_output = softmax(layer_6)
        # cost = ((Y-layer_6_output)**2).sum()

        # Negative Log Likelihood
        output = tensor.argmax(layer_6_output,axis=1)
        cost = -tensor.mean(tensor.log(layer_6_output)[tensor.arange(Y.shape[0]), Y])

        error = tensor.mean(tensor.neq(output, Y))

        parameters = [W1,b1,W2,b2,W3,b3,W5,W6,b6]

        updates = self.GradientDescent(cost,parameters)

        params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W5": W5, "W6": W6, "b6": b6}
        self.parameters = theano.function([],params)
        self.train = theano.function([X, Y], cost,updates=updates)
        self.test = theano.function([X, Y], error)
        self.predict = theano.function([X],layer_6_output)

    def GradientDescent(self, cost, parameters, learning_rate=0.011, rho=0.8, epsilon=1e-6):
        gradients = tensor.grad(cost, parameters)
        updates = []
        for param, grad in zip(parameters, gradients):
            # acc = theano.shared(param.get_value() * 0.)
            # acc_new = rho * acc + (1 - rho) * grad ** 2
            # gradient_scaling = tensor.sqrt(acc_new + epsilon)
            # grad = grad / gradient_scaling
            # updates.append((acc, acc_new))
            updates.append((param, param - learning_rate * grad))
        return updates
