import scipy.io.wavfile
from scipy import *
from scipy import signal
from os import listdir
from os.path import isfile, join
import sys


def transform(data, fs):
    if fs * 3 < len(data):
        sig = [data[i] for i in range(fs, fs * 3)]
    else:
        sig = data

    s_fft = abs(fft(sig))
    data = list()
    freq_tab = range(len(s_fft) // 2)
    for i in freq_tab:
        data.append(s_fft[i])
        if i < 200 or i > 4000:
            data[i] = 0

    return data, freq_tab


def frequency(data, freq_tab):
    filtr = list()
    wynik = data.copy()
    filtr.append(data)
    for i in range(1, 8):
        filtr.append(scipy.signal.decimate(data, i))
        for x, y in enumerate(filtr[i]):
            wynik[x] = wynik[x] * y

    wynik = [0 if i < 1 else i for i in wynik]
    return freq_tab[argmax(wynik)]


def check(file, successful=0):
    fs, data = scipy.io.wavfile.read(file)
    if len(data.shape) > 1:
        data = data.T[0]
    data, freq = transform(data, fs)

    if frequency(data, freq) > 350:
        if file[-5] == 'K':
            successful += 1
        return 'K', successful
    else:
        if file[-5] == 'M':
            successful += 1
        return 'M', successful


def check_all():
    onlyfiles = [f for f in listdir("trainall/") if isfile(join("trainall/", f))]
    successful = 0
    for file in onlyfiles:
        wynik, successful = check("trainall/" + file, successful)
    print('Success rate of gender recognition: ' + str(100*successful/len(onlyfiles))+'%')


def check_one():
    if len(sys.argv) < 2:
        print('Brak pliku')
    wynik, _ = check(sys.argv[1])
    print(wynik)


if __name__ == '__main__':
    check_one()
