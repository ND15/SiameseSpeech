from tensorflow import keras
import tensorflow as tf
from keras.layers import Conv2D, Dense, Concatenate, Reshape, ReLU, Conv1D, LeakyReLU, Lambda
from keras.layers import Input, BatchNormalization, MaxPool2D, GlobalAvgPool2D, Flatten, Dropout
from keras.models import Sequential, Model
import keras.backend as k
import tensorflow_addons as tfa


class ContrastiveLoss(keras.losses.Loss):
    def __init__(self, margin=1, **kwargs):
        super(ContrastiveLoss, self).__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        squared_pred = k.square(y_pred)
        squared_margin = k.square(k.maximum(self.margin - y_pred, 0))
        loss = k.mean(y_true * squared_pred + (1 - y_true) * squared_margin)
        return loss

    def get_config(self):
        base_config = super(ContrastiveLoss, self).get_config()
        return {**base_config, "margin": self.margin}


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = k.sum(k.square(featsA - featsB), axis=1,
                       keepdims=True)
    # return the euclidean distance between the vectors
    return k.sqrt(k.maximum(sumSquared, k.epsilon()))


def build_siamese_network(image_size=(128, 400)):
    resnet_embeddings = Sequential([
        Reshape((128, 400, 1)),
        Conv2D(3, 3, 1, padding="SAME"),
        keras.applications.resnet.ResNet50(
            include_top=False, input_shape=(128, 400, 3),
            weights=None,
        ),
        GlobalAvgPool2D(),
    ], name="embeddings")

    for layer in resnet_embeddings.get_layer("resnet50").layers:
        layer.trainable = True

    input_1 = Input((128, 400), name="input_1")
    input_2 = Input((128, 400), name="input_2")

    feature_embeddings_1 = resnet_embeddings(input_1)
    feature_embeddings_2 = resnet_embeddings(input_2)

    concat = Concatenate()([feature_embeddings_1, feature_embeddings_2])

    dense = Dense(64, activation='relu')(concat)

    output = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[input_1, input_2], outputs=output)

    return model


def conv_blocK(filters):
    return Sequential([
        Conv2D(filters=filters, kernel_size=3, strides=2),
        BatchNormalization(),
        LeakyReLU(0.2)
    ])


def custom_build_siamese_network(filters=64):
    resnet_embeddings = Sequential([
        Reshape((80, 256, 1)),
        MBBlock(4, 32, strides=(2, 2)),
        MBBlock(4, 64, strides=(2, 2)),
        MBBlock(4, 128, strides=(2, 2)),
        # MBBlock(4, 128, strides=(2, 2)),
        keras.layers.GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
    ], name="embeddings")

    input_1 = Input((80, 256), name="input_1")
    input_2 = Input((80, 256), name="input_2")

    feature_embeddings_1 = resnet_embeddings(input_1)
    feature_embeddings_2 = resnet_embeddings(input_2)

    print(resnet_embeddings.summary())

    # concat = Lambda(euclidean_distance)([feature_embeddings_1, feature_embeddings_2])

    concat = Concatenate()([feature_embeddings_1, feature_embeddings_2])

    output = Dense(1, activation='sigmoid')(concat)

    model = Model(inputs=[input_1, input_2], outputs=output)

    return model


class MBBlock(keras.layers.Layer):
    def __init__(self, bands=4, filters=32, axis=1, kernel_size=(3, 5), strides=(1, 2), padding="SAME", **kwargs):
        super(MBBlock, self).__init__(**kwargs)
        self.bands = bands
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.axis = axis

        self.layers = [
            Conv2D(filters=self.filters, kernel_size=self.kernel_size[0], strides=self.strides[0],
                   padding=self.padding, kernel_initializer="glorot_uniform", kernel_regularizer=keras.regularizers.l2(0.01)),
            tfa.layers.InstanceNormalization(),
            # keras.layers.BatchNormalization(),
            ReLU(),

            Conv2D(filters=self.filters * 2, kernel_size=self.kernel_size[1], strides=self.strides[1],
                   padding=self.padding, kernel_initializer="glorot_uniform", kernel_regularizer=keras.regularizers.l2(0.01)),
            # keras.layers.BatchNormalization(),
            tfa.layers.InstanceNormalization(),
            ReLU(),
        ]

    def call(self, inputs):
        x = inputs
        splits = tf.split(x, num_or_size_splits=self.bands, axis=self.axis)

        for i in range(self.bands):
            for layer in self.layers:
                splits[i] = layer(splits[i])

        return tf.concat(splits, axis=self.axis)

    def get_config(self):
        config = super(MBBlock, self).get_config()
        config.update({
            'bands': self.bands,
            'axis': self.axis,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
        })
