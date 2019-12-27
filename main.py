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
    # median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    # mode = freq[amp.argmax()]
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    IQR = Q75 - Q25
    # z = amp - amp.mean()
    # w = amp.std()
    # skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
    # kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4
    # result_d = {
    #     'mean': mean,
    #     'sd': sd,
    #     'median': median,
    #     'mode': mode,
    #     'Q25': Q25,
    #     'Q75': Q75,
    #     'IQR': IQR,
    #     'skew': skew,
    #     'kurt': kurt
    # }
    return Q25, IQR, sd


def main():
    onlyfiles = [f for f in listdir("trainall/") if isfile(join("trainall/", f))]
    successful = 0

    for file in onlyfiles:
        fs, data = scipy.io.wavfile.read("trainall/" + file)
        if len(data.shape) > 1:
            data = data[:, 0]
        t = np.arange(1024, len(data), 256)
        sum = 0
        fft = []
        freq = range(80, 256)

        for i in range(len(t)):

            data_slice = data[(t[i]-1024):t[i]]
            pre_fft = np.fft.fft(data_slice)
            fft = abs(pre_fft[40:256])

            sum += np.argmax(fft)

        q25, iqr, sd = spectral_properties(data, fs)
        result = (sum/len(t))
        print(file + " : " + str(result))
        r_threshold = 60.0
        i_threshold = 100
        if (result < r_threshold or (iqr-sd) < i_threshold) and file[4] == 'M':
            successful += 1

        if (result > r_threshold or (iqr-sd) > i_threshold) and file[4] == 'K':
            successful += 1
    print(str(100*successful/len(onlyfiles))+'%')


if __name__ == '__main__':
    main()
