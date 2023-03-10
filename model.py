import glob
import numpy as np
import pandas as pd
import argparse

from models.model_utils import build_siamese_network, ContrastiveLoss, custom_build_siamese_network
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils.preprocess import map_labels, prepare_audio_and_labels, create_dataset, create_pairs, dataset

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_dir', type=str,
                    default="D:/Downloads/Vox/vox_custom/",
                    required=False, help='Path to dataset')
parser.add_argument('--path_to_csv', type=str, default="D:/Downloads/Vox/vox_custom/vox1_meta.csv",
                    required=False, help='Path to csv')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
args = vars(parser.parse_args())

if __name__ == "__main__":
    le = LabelEncoder()

    filenames = glob.glob(args['path_to_dir'] + "**/**/*.wav")

    df = pd.read_csv(args['path_to_csv'], sep='\t')

    X_pairs, y_pairs = dataset(filenames, df)

    print(X_pairs.shape, y_pairs.shape)

    X_train_pairs, X_test_pairs, y_train_pairs, y_test_pairs = train_test_split(X_pairs, y_pairs, test_size=0.2)

    siamese = custom_build_siamese_network()
    siamese.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="sgd")
    siamese.summary()

    siamese.fit(x=[X_train_pairs[:, 0, :, :], X_train_pairs[:, 1, :, :]], y=y_train_pairs,
                batch_size=args['batch_size'], epochs=5,
                validation_data=([X_test_pairs[:, 0, :, :], X_test_pairs[:, 1, :, :]], y_test_pairs))
