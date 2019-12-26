import pandas as pd
import matplotlib.pyplot as plt
import sys
import scipy.io.wavfile
import numpy as np
import scipy.signal as scs
from os import listdir
from os.path import isfile, join


def main(audio):
    onlyfiles = [f for f in listdir("trainall/") if isfile(join("trainall/", f))]
    kobiety = []
    mezczyzni = []
    for file in onlyfiles:
        # file = "trainall/003_K.wav"
        fs, audio = scipy.io.wavfile.read("trainall/"+file)
        if len(audio.shape) > 1:
            a = audio.T[0]
        else:
            a = audio
        alen = int(len(a) / 2 - 1)
        k = np.arange(alen)
        a = a[::10]
        fft = np.fft.fft(a)
        fft = fft[:alen]
        fft = abs(fft / alen * 10)
        size = fft.size
        fft2 = list(zip(fft, k))
        fft2.sort(key=lambda x: x[0])
        if file[-5] == 'K':
            kobiety.append([fft2[-1][0], fft2[-1][1]])
        else:
            mezczyzni.append([fft2[-1][0], fft2[-1][1]])
    kobiety = np.array(kobiety)
    mezczyzni = np.array(mezczyzni)

    plt.scatter(kobiety[:, 1], kobiety[:, 0], marker="x")
    plt.scatter(mezczyzni[:, 1], mezczyzni[:, 0], marker="x")
    plt.yscale("log")
    plt.ylabel("częstotliwość (log)")
    # plt.show()
    plt.legend(("kobiety", "mężczyźni"))
    plt.savefig("korelacja.png")



if __name__ == '__main__':
    main(sys.argv[1])
