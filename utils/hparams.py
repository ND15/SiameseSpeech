class HParams(object):
    """ Hparams was removed from tf 2.0alpha so this is a placeholder
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


hparams = HParams(
    win_length=2048,
    n_fft=2048,
    hop_length=256,
    ref_level_db=50,
    min_level_db=-100,
    # mel scaling
    num_mel_bins=80,
    samplerate=16000,
    # inversion
    power=1.5,  # for spectral inversion
    griffin_lim_iters=50,
    pad=True,
)
