import pandas as pd
import matplotlib.pyplot as plt
import sys
import scipy.io.wavfile
import numpy as np
import scipy.signal as scs
from os import listdir
from os.path import isfile, join


def spectral_properties(y: np.ndarray, fs: int) -> dict:
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1 / fs)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    mean = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    amp_cumsum = np.cumsum(amp)
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    mode = freq[amp.argmax()]
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    IQR = Q75 - Q25
    z = amp - amp.mean()
    w = amp.std()
    skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
    kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4
    result_d = {
        'mean': mean,
        'sd': sd,
        'median': median,
        'mode': mode,
        'Q25': Q25,
        'Q75': Q75,
        'IQR': IQR,
        'skew': skew,
        'kurt': kurt
    }
    return result_d


def main(audio):
    onlyfiles = [f for f in listdir("trainall/") if isfile(join("trainall/", f))]

    whats = [('mean', 'sd'), ('mean', 'IQR'), ('mean', 'Q25'), ('sd', 'IQR'), ('IQR', 'Q25'), ('sd', 'Q25')]
    for what in whats:
        kobiety = []
        mezczyzni = []
        for file in onlyfiles:
            # file = "trainall/003_K.wav"
            fs, audio = scipy.io.wavfile.read("trainall/" + file)
            if len(audio.shape) > 1:
                a = audio.T[0]
            else:
                a = audio
            result = spectral_properties(a, fs)
            if file[-5] == 'K':
                kobiety.append([result[what[0]], result[what[1]]])
            else:
                mezczyzni.append([result[what[0]], result[what[1]]])
        kobiety = np.array(kobiety)
        mezczyzni = np.array(mezczyzni)

        plt.scatter(kobiety[:, 0], kobiety[:, 1], marker="x")
        plt.scatter(mezczyzni[:, 0], mezczyzni[:, 1], marker="x")

        # plt.yscale("log")
        plt.xlabel(what[0])
        plt.ylabel(what[1])

        plt.legend(("kobiety", "mężczyźni"))
        # plt.show()
        plt.savefig(what[0]+'_'+what[1]+'.png')
        plt.close('all')


if __name__ == '__main__':
    main(sys.argv[1])
