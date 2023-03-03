import glob
import numpy as np
import sys
from models.model_utils import build_siamese_network, ContrastiveLoss
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.preprocess import map_labels, prepare_audio_and_labels, create_dataset, create_pairs

if __name__ == "__main__":
    le = LabelEncoder()
    filenames = glob.glob("D:/Downloads/Vox/vox1_indian/content/vox_indian/**/**/*.wav")
    df = pd.read_csv("D:/Downloads/Vox/vox1_meta.csv", sep='\t')

    audio_files, labels = prepare_audio_and_labels(filenames)
    names = map_labels(labels, df)

    random_indices = np.random.choice(4857, 50, replace=False)

    audio_files = np.asarray(audio_files)
    names = np.asarray(names)

    mels, ids = create_dataset(audio_files[random_indices], names[random_indices])
    print(mels.shape, ids.shape)
    ids = le.fit_transform(ids)
    X_pairs, y_pairs = create_pairs(mels, ids)
    print(X_pairs.shape, y_pairs.shape)

    siamese = build_siamese_network()
    siamese.compile(loss="binary_crossentropy", metrics=["accuracy"])

    siamese.fit(x=[X_pairs[:, 0, :, :], X_pairs[:, 1, :, :]], y=y_pairs, batch_size=3, epochs=5, )
