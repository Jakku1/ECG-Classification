from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import scipy.io as spio
from scipy import stats
from scipy.signal import find_peaks
from scipy.signal import medfilt
import matplotlib.pyplot as plt
# Import scipy.special for the sigmoid function expit()
import scipy.special
import numpy
import numpy as np


def import_matlab_data():
    # use the scipy library to import the MATLAB training data file and squeeze the data
    mat = spio.loadmat('training.mat', squeeze_me=True)

    # assign the imported data to variables
    # d is a time domain recording of the recorded signal from the electrode
    # Index is the index position of each neuron spike
    # Class is the type of neuron that produced the spike - there are 4 different neurons
    d = mat['d']
    Index = mat['Index']
    Class = mat['Class']

    # return these variables so they can be used in the main body
    return d, Index, Class


def get_peaks(d):
    # flatten the overall shape of the input data by subtracting the mean
    d_shape = medfilt(d, kernel_size=201)
    d_flat = d - d_shape

    # use a median filter to smooth the curve, making it easier to detect the true peaks in the signal
    d_filt = medfilt(d_flat, kernel_size=7)

    # calculate the median absolute deviation and set the thresold accordingly
    mad = stats.median_abs_deviation(d)
    threshold = 5 * mad

    # find all potential peaks in the data above the threshold
    peaks, _ = find_peaks(d_filt, height=threshold, distance=10, width=3)

    return peaks



def split_data(d, Class, Index):

    # combine class and index vectors so index and its class cna be grouped.
    combined_vectors = np.column_stack((Index, Class))

    # sort the new matrix by the index in ascending order
    sorted_vectors = sorted(combined_vectors, key=lambda l: l[0])

    # split and flatten the matrix back into ordered vectors.
    sorted_vectors = np.transpose(sorted_vectors)
    sorted_Index, sorted_Class = np.vsplit(sorted_vectors, 2)
    sorted_Index = sorted_Index.flatten()
    sorted_Index = sorted_Index.tolist()
    sorted_Class = sorted_Class.flatten()
    sorted_Class = sorted_Class.tolist()

    # calculate length of data and find the splitting point between the train and test data
    data_length = len(Index)
    test_length = round(0.8 * data_length)

    # split all the data into seperate test and train vectors
    Index_train = sorted_Index[:test_length]
    Index_test = sorted_Index[test_length:]

    Class_train = sorted_Class[:test_length]
    Class_test = sorted_Class[test_length:]

    data_split = sorted_Index[test_length]

    d_train = d[:data_split]
    d_test = d[data_split:]

    return d_train, d_test, Class_train, Class_test, Index_train, Index_test, test_length


def get_test_spikes(Index, d):
    # add range of points either end of the peak to the array to store a map of the peak
    for x in Index:
        test_spikes.append(d[x - 10:x + 40])


def get_train_spikes(Index, d):
    for x in Index:
        # add range of points either end of the peak to the array to store a map of the peak
        train_spikes.append(d[x - 10:x + 40])


def MLP_train(input, labels):
    # Train the mlp using backpropogation
    # a network structure of 50 input nodes, 20 and 10 hidden layer nodes and 4 output nodes is used
    # learning rate is set 0.01
    clf = MLPClassifier(hidden_layer_sizes=(33, 24),
                        activation='tanh',
                        solver='adam',
                        learning_rate='constant',
                        learning_rate_init=0.01,
                        max_iter=1000,
                        shuffle=True,
                        random_state=1,
                        verbose=False)

    # fit the data to the appropriate labels
    clf.fit(input, labels)
    return clf


def MLP_test(clf, input, labels):
    # input test spikes to get classifications
    predictions = clf.predict(input)

    # produce a confusion matrix for this result
    score = confusion_matrix(predictions, labels)
    # calculate the accuracy and f1score for this result based off the true/false positive/negatives
    accuracy = accuracy_score(predictions, labels)
    f1score = f1_score(predictions, labels, average='micro')

    return score, accuracy, f1score


# initiate empty matricies
test_spikes = []
train_spikes = []
test_peaks = []

# get data from provided MATLAB file
d, Index, Class = import_matlab_data()
# split data into test and train data
d_train, d_test, Class_train, Class_test, Index_train, Index_test, test_length = split_data(d, Class, Index)

# call functions to get data for each spike
# train_peaks = get_peaks(d_train)
peaks = get_peaks(d)

for x in peaks:
    if x > Index_test[0]:
        test_peaks.append(x)

get_train_spikes(Index_train, d_train)
get_test_spikes(Index_test, d)

# train the mlp with the training data
clf = MLP_train(train_spikes, Class_train)

# test the performance of the MLP
score, accuracy, f1score = MLP_test(clf, test_spikes, Class_test)

# print the output scores to the user
print("score", score)
print("accuracy", accuracy)
print("f1 score", f1score)

