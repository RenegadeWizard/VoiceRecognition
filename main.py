import scipy.io.wavfile
import librosa
import noisereduce as nr
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn import neural_network
import xgboost


def spectral_properties(y: np.ndarray, fs: int):
    spec = np.abs(np.fft.fft(y))
    freq = np.fft.fftfreq(len(y), d=1 / fs)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    mean = (freq * amp).sum() / 1000    # KHz
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    amp_cumsum = np.cumsum(amp)
    q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1] / 1000  # KHz
    q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1] / 1000  # KHz
    iqr = q75 - q25
    return q25, iqr, sd, q75, mean


def denoise(data):
    avg = sum(abs(data)) / len(data)
    avg = avg / 10
    noise = data
    noise = noise[noise < avg]
    noise = noise[noise > -avg]
    data = nr.reduce_noise(audio_clip=data, noise_clip=noise)
    return data


def normalize(tab):
    t = [float(i) for i in tab]
    t = np.array(t)
    m = np.max(t)
    t = t / m
    return tab


def mean(fft):
    fft2 = sorted(fft)
    return sum(fft2) / len(fft2)


def q25(fft):
    fft2 = sorted(fft)
    fft2 = np.array(fft2)
    return fft2[len(fft2) // 4]

def q75(fft):
    fft2 = sorted(fft)
    fft2 = np.array(fft2)
    return fft2[len(fft2) // 4 * 3]


def main():
    onlyfiles = [f for f in listdir("trainall/") if isfile(join("trainall/", f))]
    # onlyfiles = ["022_K.wav"]
    successful = 0
    wyniki = []
    for file in onlyfiles:
        fs, data = scipy.io.wavfile.read("trainall/" + file)
        if len(data.shape) > 1:
            data = data.T[0]
        fft = np.fft.fft(data)
        alen = fft.size // 2
        fft = abs(fft[:alen]/alen*10)

        print(mean(fft), q25(fft), q75(fft), q75(fft)-q25(fft))
        wyniki.append([mean(fft), q25(fft), q75(fft), q75(fft)-q25(fft), 'male' if file[4] == 'M' else 'female'])

    wyniki = np.array(wyniki)
    data_frame = pd.DataFrame(
        {'meanFreq': normalize(wyniki[:, 0]), 'q25': normalize(wyniki[:, 1]), 'q75': normalize(wyniki[:, 2]),
         'iqr': normalize(wyniki[:, 3]), 'gender': wyniki[:, 4]})
    data_frame.to_csv("data.csv", index=False)

        # plt.subplot(1, 2, 2)
        # plt.plot(fft)
        # plt.show()

        # q25, iqr, sd, q75, mean = spectral_properties(data, fs)
        # result = (_sum/len(t))
        # result = sum(data) / len(data)
        # print(file + ", FreqResult: " + str(result) + ", IQR: " + str(iqr) + ", SD: " + str(sd))
        # print(file + ": ", end="")
    #     r_threshold = 60.0
    #     i_threshold = 300
    #     gender = None
    #     if result < r_threshold or (iqr-sd) < i_threshold:
    #         gender = 'M'
    #         print('Recognized: Male\n')
    #
    #     elif result > r_threshold or (iqr-sd) > i_threshold:
    #         gender = 'K'
    #         print('Recognized: Female\n')
    #     else:
    #         # print('FAILED TO RECOGNIZE GENDER!\n')
    #         pass
    #
    #     if file[4] == 'M':
    #         wyniki.append([result, q25, iqr, q75, mean, 'male'])
    #     else:
    #         wyniki.append([result, q25, iqr, q75, mean, 'female'])
    #     if gender == file[4]:
    #         successful += 1
    #     else:
    #         # print(file)
    #         pass
    # print('Success rate of gender recognition: ' + str(100*successful/len(onlyfiles))+'%')
    #
    # wyniki = np.array(wyniki)
    # data_frame = pd.DataFrame({'meanFunc': normalize(wyniki[:, 0]), 'q25': normalize(wyniki[:, 1]), 'iqr': normalize(wyniki[:, 2]), 'q75': normalize(wyniki[:, 3]), 'meanFreq': normalize(wyniki[:, 4]), 'gender': wyniki[:, 5]})
    #
    # data_frame.to_csv(path_or_buf='data.csv', index=False)


def process_data():
    voice = pd.read_csv('data.csv')
    le = preprocessing.LabelEncoder()
    voice["gender"] = le.fit_transform(voice['gender'])
    plt.subplots(2, 2, figsize=(10, 10))
    for i in range(1, 5):
        plt.subplot(2, 2, i)
        plt.title(voice.columns[i - 1])
        sns.kdeplot(voice.loc[voice['gender'] == 0, voice.columns[i - 1]], color='green', label='F')
        sns.kdeplot(voice.loc[voice['gender'] == 1, voice.columns[i - 1]], color='blue', label='M')
    plt.savefig("japierdole.png")

    # train, test = train_test_split(voice, test_size=0.3)
    # x_train = train.iloc[:, :-1]
    # y_train = train["gender"]
    # x_test = test.iloc[:, :-1]
    # y_test = test["gender"]
    # x_train3 = train[["meanFunc", "iqr", "q25"]]
    # y_train3 = train["gender"]
    # x_test3 = test[["meanFunc", "iqr", "q25"]]
    # y_test3 = test["gender"]
    #
    # k = knn_error(21, x_train3, y_train3, x_test3, y_test3)
    # n = dt_error(15,x_train3,y_train3,x_test3,y_test3)
    # pruned_tree = tree.DecisionTreeClassifier(criterion='gini', max_leaf_nodes=n)
    # classify(pruned_tree, x_train, y_train, x_test, y_test)


def classify(model, x_train, y_train, x_test, y_test):
    from sklearn.metrics import classification_report
    target_names = ['female', 'male']
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))


def knn_error(k, x_train, y_train, x_test, y_test):
    error_rate = []
    K = range(1, k)
    for i in K:
        knn = neighbors.KNeighborsClassifier(n_neighbors = i)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        error_rate.append(np.mean(y_pred != y_test))
    kloc = error_rate.index(min(error_rate))
    print("Lowest error is %s occurs at k=%s." % (error_rate[kloc], K[kloc]))

    plt.plot(K, error_rate, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()
    return K[kloc]


def dt_error(n,x_train,y_train,x_test,y_test):
    nodes = range(2, n)
    error_rate = []
    for k in nodes:
        model = tree.DecisionTreeClassifier(max_leaf_nodes=k)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        error_rate.append(np.mean(y_pred != y_test))
    kloc = error_rate.index(min(error_rate))
    print("Lowest error is %s occurs at n=%s." % (error_rate[kloc], nodes[kloc]))
    plt.plot(nodes, error_rate, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.xlabel('Tree Size')
    plt.ylabel('Cross-Validated MSE')
    plt.show()
    return nodes[kloc]


if __name__ == '__main__':
    # main()
    process_data()
