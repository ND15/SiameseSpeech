import glob
import itertools

import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utils.audio_utils import MelSpec
from hparams import hparams


def prepare_audio_and_labels(files, dataframe):
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


def get_length(audio_files):
    audio_lens = []
    for i in audio_files:
        audio, sr = sf.read(i)
        audio_lens.append(len(audio))
        print(i)
    return np.asarray(audio_lens)


def create_dataset(files, labels):
    mel = MelSpec(hparams)
    total_len = 102200

    mels = []
    ids = []

    for file_name, label in zip(files, labels):
        data, samplerate = sf.read(file_name)
        if len(data) < total_len:
            data = np.pad(data, (0, total_len - len(data)), 'constant', constant_values=(0, 0))
        else:
            data = data[:total_len]

        mel_spectrogram = mel.mel_spectrogram(data.astype('float32'))
        mels.append(mel_spectrogram.numpy().T)
        ids.append(label)
        print(f"Processed {file_name}")

    return np.asarray(mels), np.asarray(ids)


def create_pairs(mels, labels):
    X_pairs, y_pairs = [], []
    tuples = [(x1, y1) for x1, y1 in zip(mels, labels)]

    for t in itertools.product(tuples, tuples):
        pair_A, pair_B = t
        mel_A, label_A = t[0]
        mel_B, label_B = t[1]

        new_label = int(label_A == label_B)

        X_pairs.append([mel_A, mel_B])
        y_pairs.append(new_label)

    X_pairs = np.asarray(X_pairs)
    y_pairs = np.asarray(y_pairs)

    return X_pairs, y_pairs


if __name__ == "__main__":
    le = LabelEncoder()
    filenames = glob.glob("D:/Downloads/Vox/vox1_indian/content/vox_indian/**/**/*.wav")
    df = pd.read_csv("D:/Downloads/Vox/vox1_meta.csv", sep='\t')

    audio_files, labels = prepare_audio_and_labels(filenames, df)
    names = map_labels(labels, df)

    random_indices = np.random.choice(4857, 10, replace=False)

    audio_files = np.asarray(audio_files)
    names = np.asarray(names)

    mels, ids = create_dataset(audio_files[random_indices], names[random_indices])
    print(mels.shape, ids.shape)
    ids = le.fit_transform(ids)
    X_pairs, y_pairs = create_pairs(mels, ids)
    print(X_pairs.shape, y_pairs.shape)
