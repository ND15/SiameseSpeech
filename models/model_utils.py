from tensorflow import keras
import tensorflow as tf
from keras.layers import Conv2D, Dense, Concatenate, Reshape, ReLU
from keras.layers import Input, BatchNormalization, MaxPool2D, GlobalAvgPool2D, Flatten
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


def build_siamese_network(image_size=(256, 400)):
    resnet_embeddings = Sequential([
        Reshape((256, 400, 1)),
        Conv2D(3, 3, 1, padding="SAME"),
        keras.applications.resnet.ResNet50(
            include_top=False, input_shape=(256, 400, 3),
            weights=None,
        ),
        GlobalAvgPool2D(),
    ], name="embeddings")

    trainable = False
    # for layer in resnet_embeddings.get_layer("resnet50").layers:
    #     if layer.name == "conv5_block1_out":
    #         trainable = True
    #         print("Updated trainable weights")
    #     layer.trainable = trainable
    for layer in resnet_embeddings.get_layer("resnet50").layers:
        layer.trainable = True

    input_1 = Input((256, 400), name="input_1")
    input_2 = Input((256, 400), name="input_2")

    feature_embeddings_1 = resnet_embeddings(input_1)
    feature_embeddings_2 = resnet_embeddings(input_2)

    concat = Concatenate()([feature_embeddings_1, feature_embeddings_2])

    dense = Dense(64, activation='relu')(concat)

    output = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[input_1, input_2], outputs=output)

    return model


if __name__ == "__main__":
    siamese = build_siamese_network()
    siamese.summary()
