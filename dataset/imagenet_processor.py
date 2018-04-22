from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class ImageNet(object):

    def __init__(self, batch_size=16):
        train_data_gen = ImageDataGenerator(shear_range=0.2,zoom_range=0.2,horizontal_flip=True,rescale=1./255)
        test_data_gen = ImageDataGenerator(rescale=1./255)

        train_generator = train_data_gen.flow_from_directory(
            'dataset/imagenet/train',
            target_size=(64, 64),
            batch_size=16,
            class_mode='categorical'
        )
        train_size = 216000
        test_size = 24000
        self.train_inputs = []
        self.train_labels = []
        self.test_inputs = None
        self.test_labels = None
        # test_generator = test_data_gen.flow_from_directory(
        #     'dataset/imagenet/val',
        #     target_size=(64,64),
        #     batch_size=240000,
        #     class_mode='categorical'
        # )
        for i in range(int(train_size/batch_size)):
            inputs, labels = train_generator.next()
            self.train_inputs.append(inputs)
            self.train_labels.append(labels.astype(int))
        for i in range(int(test_size/batch_size)):
            inputs, labels = train_generator.next()
            if self.test_inputs is None:
                self.test_inputs = inputs
                self.test_labels = labels.astype(int)
            else:
                self.test_inputs = np.append(self.test_inputs, inputs, axis=0)
                self.test_labels = np.append(self.test_labels, labels.astype(int), axis=0)
