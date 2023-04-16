import io
import urllib

import librosa
import librosa.display
import soundfile as sf
import tensorflow as tf
import numpy as np
from .hparams import hparams
import matplotlib.pyplot as plt


class MelSpec:
    def __init__(self, hp):
        self.frame_length = hp.win_length
        self.fft_length = hp.n_fft
        self.frame_step = hp.hop_length
        self.ref_level_dB = hp.ref_level_db
        self.min_level_dB = hp.min_level_db
        self.num_mel_bins = hp.num_mel_bins
        self.samplerate = hp.samplerate,
        self.mel_lower_edge_hertz = 0,
        self.mel_upper_edge_hertz = int(self.samplerate[0] / 2),
        self.power = hp.power
        self.griffin_lim_iters = hp.griffin_lim_iters
        self.padding = hp.pad

        self.mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=int(self.fft_length / 2) + 1,
            sample_rate=self.samplerate[0],
            lower_edge_hertz=self.mel_lower_edge_hertz[0],
            upper_edge_hertz=self.mel_upper_edge_hertz[0],
            dtype=tf.dtypes.float32,
            name=None,
        )

    def normalize(self, S):
        return tf.clip_by_value(
            (S - self.min_level_dB) / -self.min_level_dB, 0, 1)

    def denormalize(self, S):
        return (tf.clip_by_value(S, 0, 1) * -self.min_level_dB) + self.min_level_dB

    @staticmethod
    def db_to_amp(S):
        return tf.math.pow(tf.ones(tf.shape(S)) * 10.0, S * 0.05)

    @staticmethod
    def tf_log_base(base, S):
        num = tf.math.log(S)
        base = tf.math.log(tf.constant(base, dtype=num.dtype))
        return num / base

    def amplitude_to_db(self, S):
        return 20 * self.tf_log_base(
            base=10, S=tf.clip_by_value(tf.abs(S), 1e-5, 1e100))

    def stft(self, waveform):
        stft = tf.signal.stft(
            signals=waveform,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length,
            window_fn=tf.signal.hann_window,
            pad_end=True,
        )
        return stft

    def spectrogram(self, waveform):
        D = self.stft(waveform)
        S = self.amplitude_to_db(tf.abs(D)) - self.ref_level_dB
        return self.normalize(S)

    def mel_spectrogram(self, waveform, normalize=True):
        spec = self.spectrogram(waveform)
        if normalize:
            mel_f = librosa.core.mel_frequencies(
                n_mels=self.num_mel_bins + 2,
                fmin=self.mel_lower_edge_hertz,
                fmax=self.mel_upper_edge_hertz,
            )

            enorm = tf.dtypes.cast(
                tf.expand_dims(tf.constant(2.0 / (mel_f[2: self.num_mel_bins + 2] - mel_f[:self.num_mel_bins])),
                               0),
                tf.float32,
            )
            self.mel_matrix = tf.multiply(self.mel_matrix, enorm[..., 0])
            self.mel_matrix = tf.divide(self.mel_matrix,
                                        tf.reduce_sum(self.mel_matrix, axis=0))
            mel_spec = tf.tensordot(spec, self.mel_matrix, 1)
            return mel_spec
        else:
            return tf.tensordot(spec, self.mel_matrix, 1)

    def invert_mel_spectrogram(self, S):
        with np.errstate(divide="ignore", invalid="ignore"):
            mel_inversion_matrix = tf.constant(
                np.nan_to_num(
                    np.divide(self.mel_matrix.numpy().T,
                              np.sum(self.mel_matrix.numpy(), axis=1))
                ).T
            )
        mel_spec_inv = tf.tensordot(S, tf.transpose(mel_inversion_matrix), 1)
        return mel_spec_inv

    def inv_spectrogram(self, S):
        return self.db_to_amp(
            self.denormalize(S) + self.ref_level_dB
        )


if __name__ == "__main__":
    mel = MelSpec(hparams)
    total_len = 102200
    url = "https://raw.githubusercontent.com/timsainb/python_spectrograms_and_inversion/master/bushOffersPeace.wav"
    response = urllib.request.urlopen(url)
    data, samplerate = sf.read(io.BytesIO(response.read()))
    # data, samplerate = librosa.load("D:/MyWork/Journal_01/UNet_Music/musdb_resampled/test/4/mixture.wav",
    #                                 sr=22050)
    print(len(data))

    # if len(data) < total_len:
    #     data = np.pad(data, (0, total_len - len(data)), 'constant', constant_values=(0, 0))
    # else:
    #     data = data[:total_len]

    mel_s = mel.mel_spectrogram(data.astype('float32'))
    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(mel_s.T, x_axis='time',
    #                                y_axis='mel', sr=22050,
    #                                fmax=16000, ax=ax)
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')
    # ax.set(title='Mel-frequency spectrogram')
    # plt.show()
    plt.matshow(mel_s.numpy().T, aspect='auto', origin='lower')
    plt.show()