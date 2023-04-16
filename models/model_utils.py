from tensorflow import keras
import tensorflow as tf
from keras.layers import Conv2D, Dense, Concatenate, Reshape, ReLU, Conv1D, LeakyReLU, Lambda
from keras.layers import Input, BatchNormalization, MaxPool2D, GlobalAveragePooling2D, Flatten, Dropout
from keras.models import Sequential, Model
import keras.backend as k


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
    (featsA, featsB) = vectors
    sumSquared = k.sum(k.square(featsA - featsB), axis=1,
                       keepdims=True)
    return k.sqrt(k.maximum(sumSquared, k.epsilon()))


def build_siamese_network(image_size=(512, 256)):
    resnet_embeddings = Sequential([
        Reshape((512, 256, 1)),
        Conv2D(3, 3, 1, padding="SAME"),
        keras.applications.resnet.ResNet50(
            include_top=False, input_shape=(512, 256, 3),
            weights=None,
        ),
        GlobalAveragePooling2D(),
    ], name="embeddings")

    for layer in resnet_embeddings.get_layer("resnet50").layers:
        layer.trainable = True

    input_1 = Input((512, 256), name="input_1")
    input_2 = Input((512, 256), name="input_2")

    feature_embeddings_1 = resnet_embeddings(input_1)
    feature_embeddings_2 = resnet_embeddings(input_2)

    concat = Concatenate()([feature_embeddings_1, feature_embeddings_2])

    dense = Dense(64, activation='relu')(concat)

    output = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[input_1, input_2], outputs=output)
    
    model.summary()

    return model


