# This code is for the purpose of spike sorting to complete the tasks outlined in coursework C
import scipy.io as spio
from scipy import stats
from scipy.signal import find_peaks
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import numpy
import numpy as np
# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
# k Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


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
    # subtract the general shape of the curve to flatten it.
    d_flat = d - d_shape

    # use a median filter to smooth the curve, making it easier to detect the true peaks in the signal
    d_filt = medfilt(d_flat, kernel_size=7)

    # calculate the median absolute deviation
    mad = stats.median_abs_deviation(d)
    # set threshold for spikes to be detected
    threshold = 5 * mad
    # identify all peaks in the data
    peaks, _ = find_peaks(d_filt, height=threshold, distance=10, width=3)

    # plot filtered signal
    plt.plot(d_filt)
    # plot the signal, identifying the peaks
    # any obviously incorrect peaks can be seen and explained
    plt.plot(peaks, d_filt[peaks], "x")
    plt.show()

    return peaks


def split_data(d, Class, Index):

    # combine class and index vectors so index and its class cna be grouped.
    combined_vectors = np.column_stack((Index, Class))

    # sort the new matrix by the index in ascending order
    sorted_vectors = sorted(combined_vectors, key=lambda l: l[0])

    #split and flatten the matrix back into ordered vectors.
    sorted_vectors = np.transpose(sorted_vectors)
    sorted_Index, sorted_Class = np.vsplit(sorted_vectors, 2)
    sorted_Index = sorted_Index.flatten()
    sorted_Index = sorted_Index.tolist()
    sorted_Class = sorted_Class.flatten()
    sorted_Class = sorted_Class.tolist()

    # calculate length of data and find the splitting point between the train and test data
    data_length = len(Index)
    test_length = round(0.8*data_length)

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
        # add spike data into a new list
        test_spikes.append(d[x - 10:x + 40])


def get_train_spikes(Index, d):
    for x in Index:
        # add spike data into a new list
        train_spikes.append(d[x - 10:x + 40])


# ------------------------------------------ RUN CODE -----------------------------------------------------

#
test_peaks = []
# array to store data points for each spike in testing
test_spikes = []
# array to store data points for each spike in training
train_spikes = []
# create 4 output target
target_array = []

# get data from provided MATLAB file
d, Index, Class = import_matlab_data()
# split data into test and train data
d_train, d_test, Class_train, Class_test, Index_train, Index_test, test_length = split_data(d, Class, Index)

# call functions to get data for each spike
peaks = get_peaks(d)

for x in peaks:
    if x > Index_test[0]:
        test_peaks.append(x)

get_train_spikes(Index_train, d_train)
get_test_spikes(Index_test, d)

# --------------------------------------------------PCA Definition------------------------------------------------------
# Define training and testing data
train_data = train_spikes
train_labels = Class_train
test_data = test_spikes
test_labels = Class_test

# Select number of components to extract
pca = PCA(n_components=7)

# Fit to the training data
pca.fit(train_data)

# Determine amount of variance explained by components
print("Total Variance Explained: ", numpy.sum(pca.explained_variance_ratio_))

# ---------------------------------------------------KNN CODE-----------------------------------------------------------
# Extract the principal components from the training data
train_ext = pca.fit_transform(train_data)

# Transform the test data using the same components
test_ext = pca.transform(test_data)

# Normalise the data sets
min_max_scaler = MinMaxScaler()
train_norm = min_max_scaler.fit_transform(train_ext)
test_norm = min_max_scaler.fit_transform(test_ext)

# Create a KNN classification system with k = 5
# Uses the p2 (Euclidean) norm
knn = KNeighborsClassifier(n_neighbors=5, p=2)
knn.fit(train_norm, train_labels)

# Feed the test data in the classifier to get the predictions
pred = knn.predict(test_norm)

print(pred)

# -------------------------------- Performance Metric ----------------------------------------------

# calculate the confusion matrix of the predicted values against the class values
score = confusion_matrix(pred, Class_test)

# calculate teh accuracy of the data
accuracy = accuracy_score(pred, Class_test)

# calculate the micro f1score using the true/false positives/negatives
f1score = f1_score(pred, Class_test, average='micro')


print("score", score)
print("accuracy", accuracy)
print("f1 score", f1score)

