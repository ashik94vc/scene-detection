import theano
from theano import tensor
import numpy as np
from theano.tensor.nnet import conv2d,relu,sigmoid
from theano.tensor.signal.pool import pool_2d
from PIL import Image
from scipy.misc import toimage

__X = tensor.tensor4()
__W1 = theano.shared(np.random.randn(6,3,9,9))
__b1 = theano.shared(np.zeros(6,).astype(theano.config.floatX))
__W2 = theano.shared(np.random.randn(32,6,9,9))
__b2 = theano.shared(np.zeros(32,).astype(theano.config.floatX))
__W3 = theano.shared(np.random.randn(64,32,9,9))
__b3 = theano.shared(np.zeros(64,).astype(theano.config.floatX))
__W5 = theano.shared(np.random.randn(64*9*9,84))
__b5 = theano.shared(np.zeros(64*9*9,))
__W6 = theano.shared(np.random.randn(84,2))
__b6 = theano.shared(np.zeros(84,))

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

model = theano.function([__X], __layer_6_output)
