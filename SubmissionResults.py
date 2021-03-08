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


def import_submission_data():
    # use the scipy library to import the MATLAB training data file and squeeze the data
    mat = spio.loadmat('submission.mat', squeeze_me=True)

    # assign the imported data to variables
    # d is a time domain recording of the recorded signal from the electrode
    d_test = mat['d']

    # return these variables so they can be used in the main body
    return d_test


def get_peaks(d):
    # flatten the overall shape of the input data by subtracting the mean
    d_shape = medfilt(d, kernel_size=201)
    d_flat = d - d_shape

    # use a median filter to smooth the curve, making it easier to detect the true peaks in the signal
    d_filt = medfilt(d_flat, kernel_size=7)
    plt.plot(d_filt)
    plt.show()


    # calculate the median absolute deviation and set the threshold accordingly
    mad = stats.median_abs_deviation(d_filt)
    threshold = 5 * mad

    # find all potential peaks in the data above the threshold
    peaks, _ = find_peaks(d_filt, height=threshold, distance=10, width=3)

    return peaks, d



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

    return sorted_Index, sorted_Class


def get_test_spikes(Index, d):
    # add range of points either end of the peak to the array to store a map of the peak
    for x in Index:
        # np.append(test_spikes, d[x - 20:x + 30], axis=0)
        test_spikes.append(d[x - 17:x + 33])


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
                        learning_rate_init=0.0003587131170877944,
                        max_iter=1000,
                        shuffle=True,
                        random_state=1,
                        verbose=False)

    # fit the data to the appropriate labels
    clf.fit(input, labels)
    return clf


def MLP_test(clf, input):
    # input test spikes to get classifications
    predictions = clf.predict(input)

    return predictions


# initiate empty matricies
test_spikes = []
# test_spikes = np.array(test_spikes)
train_spikes = []
test_peaks = []

# get data from provided MATLAB file
d, Index, Class = import_matlab_data()
# get submission data from matlab
d_test = import_submission_data()
# split data into test and train data
sorted_Index, sorted_Class = split_data(d, Class, Index)

# call functions to get data for each spike
# train_peaks = get_peaks(d_train)
peaks, d_test_flat = get_peaks(d_test)
peaks = peaks[:-1]

get_train_spikes(sorted_Index, d)
get_test_spikes(peaks, d_test_flat)

count1 = 0
count2 = 0
count3 = 0
count4 = 0

# train the mlp with the training data
clf = MLP_train(train_spikes, sorted_Class)

pred_Class = MLP_test(clf, test_spikes)

plt.figure()
for index, record in enumerate(pred_Class):
    if record == 1:
        plt.plot(test_spikes[index])
        count1 += 1


plt.show()

plt.figure()
for index, record in enumerate(pred_Class):
    if record == 2:
        plt.plot(test_spikes[index])
        count2 += 1

plt.show()

plt.figure()
for index, record in enumerate(pred_Class):
    if record == 3:
        plt.plot(test_spikes[index])
        count3 += 1

plt.show()

plt.figure()
for index, record in enumerate(pred_Class):
    if record == 4:
        plt.plot(test_spikes[index])
        count4 += 1

plt.show()

print("class 1:", count1, "class 2:", count2, "class 3:", count3, "class 4:", count4)


mdic = {"Class": pred_Class, "Index": peaks}

scipy.io.savemat('MLP_results.mat', mdic)

