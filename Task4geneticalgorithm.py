from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import scipy.io as spio
from scipy import stats
from scipy.signal import find_peaks
from scipy.signal import medfilt
from random import randint, random
from operator import add
from functools import reduce
import matplotlib.pyplot as plt
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

    # calculate the median absolute deviation and set the threshold accordingly
    mad = stats.median_abs_deviation(d)
    threshold = 5 * mad
    # find all of the peaks in the data
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


def MLP_train(input, labels, h1nodes, h2nodes, lrate):
    # Train the mlp using backpropogation
    # a network structure of 50 input nodes, 20 and 10 hidden layer nodes and 4 output nodes is used
    # learning rate is set 0.01
    clf = MLPClassifier(hidden_layer_sizes=(h1nodes, h2nodes),
                        activation='tanh',
                        solver='adam',
                        learning_rate='constant',
                        learning_rate_init=lrate,
                        max_iter=1000,
                        shuffle=True,
                        random_state=1,
                        verbose=False)
    # print(h1nodes, "\n", h2nodes, "\n", lrate)
    # fit train data to the labels
    clf.fit(input, labels)
    return clf


def MLP_test(clf, input, labels):
    # input the test data to find outputs
    predictions = clf.predict(input)
    # calculate confusion matricies and performance metrics using true/false positives/negatives
    score = confusion_matrix(predictions, labels)
    accuracy = accuracy_score(predictions, labels)
    f1score = f1_score(predictions, labels, average='micro')

    return score, accuracy, f1score

def individual(h1min, h1max, h2min, h2max, lrmin, lrmax):
    'Create a member of the population.'
    return [randint(h1min, h1max), randint(h2min, h2max), np.random.random()]


def population(count, h1min, h1max, h2min, h2max, lrmin, lrmax):
    """
    Create a number of individuals (i.e. a population). consisting of three values

    count: the number of individuals in the population
    h1min & h2min: minimum possible value of hidden layer nodes 1 and 2
    h1max & h2max: the maximum possible value of hidden layer nodes 1 and 2
    lrmin & lrmax: the minimum and maximum learning rate values

    """
    return [individual(h1min, h1max, h2min, h2max, lrmin, lrmax) for x in range(count)]


def fitness(individual, target, input, labels, test_spikes, Class_test):
    """
    Determine the fitness of an individual. numbers use sign and magnitude. Parameters are taken from the individual
    and fed into the MLP as input parameters. The network is trained and then the test data is input to test the
    performance. the f1score is compared against a target of 1 which is the ideal value of a system with no errors.

    individual: the individual to evaluate
    target: the intended value of the fitness funciton
    input: training spikes
    labels: training class labels
    test_spikes: the test spikes
    Class_test: the class labels for the test spikes
    """

    # split individual into the parameters
    h1nodes = individual[0]
    h2nodes = individual[1]
    learning_rate = individual[2]

    # train the MLP
    clf = MLP_train(input, labels, h1nodes, h2nodes, learning_rate)

    # calculate the score of the function
    score, accuracy, f1score = MLP_test(clf, test_spikes, Class_test)

    # calculate the fitness of the function
    fitness_value = abs(target - f1score)
    return fitness_value


def grade(pop, target, train_spikes, Class_train, test_spikes, Class_test):
    'Find average fitness for a population.'
    summed = reduce(add, (fitness(x, target, train_spikes, Class_train, test_spikes, Class_test) for x in pop))
    return summed / (len(pop) * 1.0)


def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01):
    graded = [(fitness(x, target, train_spikes, Class_train, test_spikes, Class_test), x) for x in pop]
    graded = [x[1] for x in sorted(graded)]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]
    # randomly add other individuals to
    # promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)
    # mutate some individuals
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, 1)
            # this mutation is not ideal, because it
            # restricts the range of possible values,
            # but the function is unaware of the min/max
            # values used to create the individuals,
            individual[pos_to_mutate] = randint(h1min, h1max)
    for individual in parents:
        if mutate > random():
            individual[2] = np.random.random()
    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length - 1)
        female = randint(0, parents_length - 1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = int(len(male) / 2)
            child = male[:half] + female[half:]
            children.append(child)
    parents.extend(children)
    return parents


# ----------------- main code -----------------

# initialise vectors for storing data
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

# define genetic algorithm parameters
target = 1
p_count = 200
h1min = 1
h1max = 100
h2min = 1
h2max = 100
lrmin = 0
lrmax = 0.1
generations = 25

# initialise a population
p = population(p_count, h1min, h1max, h2min, h2max, lrmin, lrmax)
# calculate fitness history
fitness_history = [grade(p, target, train_spikes, Class_train, test_spikes, Class_test)]
# evolve populations according to the number of generations
for i in range(generations):
    p = evolve(p, target)
    grade_p = grade(p, target, train_spikes, Class_train, test_spikes, Class_test)
    fitness_history.append(grade_p)
    print(p)

# print error to target value
for datum in fitness_history:
    print(datum)

# plot the fitness over time for the function
plt.plot(fitness_history)
plt.show()

