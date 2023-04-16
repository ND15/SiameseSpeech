import glob
import itertools
import math
from abc import ABC

import pandas as pd
import soundfile as sf
import tensorflow as tf
import numpy as np
from models.model_utils import ContrastiveLoss, build_siamese_network
from utils.audio_utils import MelSpec
from utils.hparams import hparams

INIT_LR = 1e-5
MAX_LR = 1e-2
BATCH_SIZE = 16
BUFFER_SIZE = 1000
AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAINING_SIZE = 1060000
VAL_SIZE = 3000


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def prepare_audio_and_labels(files):
    audio_files = []
    labels = []

    for i in files:
        audio_files.append(i)
        labels.append(i.split('\\')[-3])

    return audio_files, labels


def map_labels(labels, dataframe):
    names = []
    df_labels = dataframe.copy()
    df_labels['VoxCeleb1 ID'] = df_labels['VoxCeleb1 ID'].astype('str')
    df_labels['VGGFace1 ID'] = df_labels['VGGFace1 ID'].astype('str')
    df_labels = df_labels[df_labels['VoxCeleb1 ID'].isin(set(labels))]
    df_labels = df_labels[['VoxCeleb1 ID', 'VGGFace1 ID']]

    for i in labels:
        values = df_labels[df_labels['VoxCeleb1 ID'] == i].values
        names.append(values[0][-1])

    return np.asarray(names)


def create_dataset(files, labels):
    mel = MelSpec(hparams)
    total_len = 65536

    mels = []
    ids = []

    for file_name, label in zip(files, labels):
        data, samplerate = sf.read(file_name)
        if len(data) < total_len:
            data = np.pad(data, (0, total_len - len(data)), 'constant', constant_values=(0, 0))
        else:
            data = data[:total_len]

        mel_spectrogram = mel.spectrogram(data.astype('float32'))
        mels.append(mel_spectrogram.numpy().T)
        ids.append(label)
        print(f"Processed {file_name}")

    return np.asarray(mels), np.asarray(ids)


def parse_single_mel(X_pairs, label):
    data = {
        'mel_filters': _int64_feature(X_pairs.shape[1]),
        'length': _int64_feature(X_pairs.shape[2]),
        'mel_spec': _bytes_feature(serialize_array(X_pairs)),
        'label': _int64_feature(label),
    }

    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out


def write_images_tfr(filenames, df, filename: str = "mels"):
    tfr_name = filename + ".tfrecords"

    options = tf.io.TFRecordOptions(tf.io.TFRecordOptions(compression_type='GZIP'))
    writer = tf.io.TFRecordWriter(tfr_name, options)
    count = 0

    df = pd.read_csv(df, sep='\t')

    filenames = glob.glob(filenames)
    audio_files, labels = prepare_audio_and_labels(filenames)

    names = map_labels(labels, df)

    print(len(audio_files))

    random_indices = np.random.choice(len(audio_files), len(audio_files), replace=False)

    mels, ids = create_dataset(np.asarray(audio_files)[random_indices],
                               np.asarray(names)[random_indices])

    tuples = [(x1, y1) for x1, y1 in zip(mels, ids)]

    for t in itertools.product(tuples, tuples):
        mel_A, label_A = t[0]
        mel_B, label_B = t[1]

        mel_A = mel_A[1:, :]
        mel_B = mel_B[1:, :]

        print(mel_A.shape, mel_B.shape)

        new_label = int(label_A == label_B)

        concat_mels = np.vstack((mel_A[np.newaxis, ...], mel_B[np.newaxis, ...]))

        out = parse_single_mel(concat_mels, label=new_label)
        writer.write(out.SerializeToString())

        count += 1

    writer.close()
    print(f"Wrote {count} images")


def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'mel_filters': tf.io.FixedLenFeature([], tf.int64),
        'length': tf.io.FixedLenFeature([], tf.int64),
        'mel_spec': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data)

    mel_filters = content['mel_filters']
    length = content['length']
    label = content['label']
    mel_spec = content['mel_spec']

    # get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(mel_spec, out_type=tf.float32)
    feature = tf.reshape(feature, shape=[2, mel_filters, length])
    return feature, label


class SiameseModel(tf.keras.Model, ABC):
    def __init__(self, siamese_model, **kwargs):
        super(SiameseModel, self).__init__(**kwargs)
        self.s_optimizer = None
        self.loss_fn = None
        self.siamese_model = siamese_model
        self.loss_metric = None
        self.acc_metric = None

    def compile(self, s_optimizer, loss_fn, **kwargs):
        super(SiameseModel, self).compile(**kwargs)
        self.loss_fn = loss_fn
        self.s_optimizer = s_optimizer
        self.loss_metric = tf.keras.metrics.Mean(name="s_loss")
        self.acc_metric = tf.keras.metrics.BinaryAccuracy()

    @property
    def metrics(self):
        return [self.loss_metric, self.acc_metric]

    def train_step(self, data):
        X_pairs, y_pairs = data
        x_1 = X_pairs[:, 0, :, :]
        x_2 = X_pairs[:, 1, :, :]

        with tf.GradientTape() as tape:
            y_pred = self.siamese_model([x_1, x_2])
            s_loss = self.loss_fn(y_pairs, y_pred)

        grads = tape.gradient(s_loss, self.siamese_model.trainable_weights)
        self.s_optimizer.apply_gradients(
            zip(grads, self.siamese_model.trainable_weights)
        )

        self.loss_metric.update_state(s_loss)
        self.acc_metric.update_state(y_pairs, y_pred)

        return {
            'mean_loss': self.loss_metric.result(),
            'accuracy': self.acc_metric.result()
        }

    def test_step(self, data):
        X_pairs, y_pairs = data
        x_1 = X_pairs[:, 0, :, :]
        x_2 = X_pairs[:, 1, :, :]

        y_pred = self.siamese_model([x_1, x_2], training=False)
        s_loss = self.loss_fn(y_pairs, y_pred)

        self.loss_metric.update_state(s_loss)
        self.acc_metric.update_state(y_pairs, y_pred)

        return {
            'mean_loss': self.loss_metric.result(),
            'accuracy': self.acc_metric.result()
        }


class SiameseModel_2(tf.keras.Model, ABC):
    def __init__(self, siamese_model, **kwargs):
        super(SiameseModel_2, self).__init__(**kwargs)
        self.siamese_model = siamese_model

    def train_step(self, data):
        X_pairs, y_pairs = data
        x_1 = X_pairs[:, 0, :, :]
        x_2 = X_pairs[:, 1, :, :]

        # tf.print("Real:", y_pairs[:128])

        with tf.GradientTape() as tape:
            y_pred = self.siamese_model([x_1, x_2], training=True)

            # tf.print("\nMax: ", tf.reduce_max(y_pred), "Min: ", tf.reduce_min(y_pred))

            loss = self.compiled_loss(y_pairs, y_pred, regularization_losses=self.losses)

        grads = tape.gradient(loss, self.siamese_model.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.siamese_model.trainable_variables)
        )

        self.compiled_metrics.update_state(y_pairs, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        X_pairs, y_pairs = data
        x_1 = X_pairs[:, 0, :, :]
        x_2 = X_pairs[:, 1, :, :]

        y_pred = self.siamese_model([x_1, x_2], training=False)

        self.compiled_loss(y_pairs, y_pred, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y_pairs, y_pred)

        return {m.name: m.result() for m in self.metrics}


def get_dataset(filename):
    dataset = tf.data.TFRecordDataset(filename, compression_type='GZIP')
    dataset = dataset.map(parse_tfr_element)

    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=False)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    mel_valid_dataset = dataset.take(VAL_SIZE)

    mel_train_dataset = dataset.take(TRAINING_SIZE)

    return mel_train_dataset, mel_valid_dataset


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print("Up to batch {}, the average loss is {:7.2f} and accuracy is {:7.2f}.".format(batch, logs["loss"],
                                                                                            logs["accuracy"]))


if __name__ == "__main__":
    # write_images_tfr("D:/Downloads/Vox/vox_custom_2/**/**/*.wav",
    #                  df="D:/Downloads/Vox/vox_custom_2/vox1_meta.csv",
    #                  filename='specs')

    train_dataset, valid_dataset = get_dataset("specs.tfrecords")

    siamese_model = build_siamese_network()

    model = SiameseModel_2(siamese_model)

    compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / BATCH_SIZE))

    val_step_per_epoch = compute_steps_per_epoch(VAL_SIZE)

    steps_per_epoch = compute_steps_per_epoch(TRAINING_SIZE)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=ContrastiveLoss(),
                  metrics=["accuracy"])

    model.fit(train_dataset, batch_size=BATCH_SIZE, epochs=1000, steps_per_epoch=200,
              validation_data=valid_dataset, validation_steps=val_step_per_epoch)

    print(model.evaluate(valid_dataset, steps=1))
