import pandas as pd
import matplotlib.pyplot as plt
import sys
import scipy.io.wavfile


def main(audio):
    fs, audio = scipy.io.wavfile.read(audio)


if __name__ == '__main__':
    main(sys.argv[1])
