from keras.preprocessing.image import ImageDataGenerator

class ImageNet(object):

    def __init__(self, batch_size=16):
        train_data_gen = ImageDataGenerator(shear_range=0.2,zoom_range=0.2,horizontal_flip=True,rescale=1./255)
        test_data_gen = ImageDataGenerator(rescale=1./255)

        train_generator = train_data_gen.flow_from_directory(
            'dataset/imagenet/train',
            target_size=(64, 64),
            batch_size=160000,
            class_mode='categorical'
        )

        # test_generator = test_data_gen.flow_from_directory(
        #     'dataset/imagenet/val',
        #     target_size=(64,64),
        #     batch_size=240000,
        #     class_mode='categorical'
        # )

        inputs, labels = train_generator.next()
        self.train_inputs = inputs[:144000]
        self.train_labels = labels[:144000]
        self.test_inputs = inputs[144000:]
        self.test_labels = inputs[144000:]
        # self.test_inputs, self.test_labels = test_generator.next()
