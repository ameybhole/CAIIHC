from keras.models import *
from keras.optimizers import Adam
from keras.layers import *
from keras.engine import Model
from keras.applications.resnet50 import ResNet50
from keras_applications.vgg16 import VGG16
from keras_vggface import VGGFace

class Models(object):
    def __init__(
            self,
            trainable,
            num_classes,
            weights,
            include_top,
            pooling,
            activation
    ):
        """

        """

        self.trainable = trainable
        self.num_classes = num_classes
        self.weights = weights
        self.include_top = include_top
        self.pooling = pooling
        self.activation = activation

        self.model_out = None

    def resnet50(self):
        """
        :return:

        """
        base_model = ResNet50(weights=self.weights, include_top=self.include_top,
                              pooling=self.pooling)
        x = base_model.layers[-1]
        out = Dense(units=self.num_classes, activation=self.activation, name='output',
                    use_bias=True)(x.output)

        self.model_out = Model(inputs=base_model.input, outputs=out)

    def facenet(self):
        """
        :return:

        """
        facenet_model = load_model('weights/facenet_keras.h5')
        facenet_model.load_weights('weights/facenet_keras_weights.h5')
        x = facenet_model.layers[-3]
        out = Dense(self.num_classes, activation=self.activation, name='output', use_bias=True)(x.output)

        self.model_out = Model(inputs=facenet_model.input, outputs=out)

    def vgg16(self):
        """
        :return:

        """
        base_model = VGG16(weights=self.weights, include_top=self.include_top,
                              pooling=self.pooling)
        x = base_model.layers[-1]
        out = Dense(self.num_classes, activation=self.activation, name='output', use_bias=True)(
            x.output)

        self.model_out = Model(inputs=VGG16.input, outputs=out)

    def vggface(self):
        """
        :return:

        """
        base_model = VGGFace(weights=self.weights, include_top=self.include_top,
                              pooling=self.pooling)
        x = base_model.layers[-1]
        out = Dense(self.num_classes, activation=self.activation, name='output', use_bias=True)(
            x.output)

        self.model_out = Model(inputs=VGGFace.input, outputs=out)

    def model_trainable(self):
        for layer in self.model_out.layers:
            layer.trainable = self.trainable

    def model_compile(
            self,
            learning_rate: float = 1e-4,
            beta1: float = 0.9,
            beta2: float = 0.999,
            epsilon: float = 1e-8,
    ):
        """
        Function to compile the model
        :param learning_rate: Initial learning rate for ADAM optimizer
        :param beta1: Exponential decay rate for the running average of the gradient
        :param beta2: Exponential decay rate for the running average of the square of the gradient
        :param epsilon: Epsilon parameter to prevent division by zero error
        :return: Compiled Keras model
        """

        adam = Adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
        self.model_out.compile(
            loss="categorical_crossentropy",
            optimizer=adam,
            metrics=["categorical_accuracy"],
        )

        return self.model_out
