import scipy.io.wavfile
import numpy as np
from os import listdir
from os.path import isfile, join


def spectral_properties(y: np.ndarray, fs: int):
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1 / fs)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    mean = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    amp_cumsum = np.cumsum(amp)
    q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    iqr = q75 - q25
    return q25, iqr, sd


def main():
    onlyfiles = [f for f in listdir("trainall/") if isfile(join("trainall/", f))]
    successful = 0

    for file in onlyfiles:
        fs, data = scipy.io.wavfile.read("trainall/" + file)
        if len(data.shape) > 1:
            data = data[:, 0]
        t = np.arange(1024, len(data), 256)
        _sum = 0

        for i in range(len(t)):
            data_slice = data[(t[i]-1024):t[i]]
            pre_fft = np.fft.fft(data_slice)
            fft = abs(pre_fft[40:256])
            _sum += np.argmax(fft)

        q25, iqr, sd = spectral_properties(data, fs)
        result = (_sum/len(t))
        print(file + ", FreqResult: " + str(result) + ", IQR-SD: " + str(iqr-sd))
        r_threshold = 60.0
        i_threshold = 100
        if (result < r_threshold or (iqr-sd) < i_threshold) and file[4] == 'M':
            successful += 1
            print('Recognized: Male\n')

        elif (result > r_threshold or (iqr-sd) > i_threshold) and file[4] == 'K':
            successful += 1
            print('Recognized: Female\n')
        else:
            print('FAILED TO RECOGNIZE GENDER!\n')

    print('Success rate of gender recognition: ' + str(100*successful/len(onlyfiles))+'%')


if __name__ == '__main__':
    main()
