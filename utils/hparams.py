class HParams(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


hparams = HParams(
    win_length=1024,
    n_fft=1024,
    hop_length=256,
    ref_level_db=50,
    min_level_db=-100,
    # mel scaling
    num_mel_bins=256,
    samplerate=22050,
    # inversion
    power=1.5,  # for spectral inversion
    griffin_lim_iters=50,
    pad=True,
)