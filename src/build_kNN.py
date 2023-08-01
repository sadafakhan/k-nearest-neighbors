import os
import sys
import numpy as np
from numpy import dot, sqrt
from numpy.linalg import norm
from collections import Counter
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

# take in parameters
training_data = sys.argv[1]
test_data = sys.argv[2]
k_val = int(sys.argv[3])
similarity_func = int(sys.argv[4])
sys_output = sys.argv[5]


# take a dataset and return a list of lists, each sublist contains the actual label & a dict representing the vector
def vectorizer(training, testing):
    # initialize counters, data structures
    feat2col = 0
    vocab = {}
    columns = {}
    training_labels = []
    testing_labels = []

    # format training data
    training_formatted = open(os.path.join(os.path.dirname(__file__), training), 'r').read().split("\n")[:-1]
    testing_formatted = open(os.path.join(os.path.dirname(__file__), testing), 'r').read().split("\n")[:-1]

    # extract the vocabulary (columns) and number of instances (rows)
    for vec in training_formatted:
        word_counts = vec.split(" ")[1:-1]
        for label in word_counts:
            feat = label.split(":")[0]

            # if an encountered word-feature isn't accounted for in the training vocabulary
            # add it into the reference dicts
            if feat not in vocab:
                vocab[feat] = feat2col
                columns[feat2col] = feat
                feat2col += 1

    # create array of training instance x word count
    training_array = np.zeros((len(training_formatted), feat2col))
    for i in range(len(training_formatted)):
        split = training_formatted[i].split(" ")
        class_label = split[0]
        word_counts = split[1:-1]

        # keep track of training labels. i-th item in actual_labels is the label for i-th row/vector in array
        training_labels.append(class_label)

        # change cell word counts
        for pair in word_counts:
            feat = pair.split(":")[0]
            count = int(pair.split(":")[1])
            training_array[i, vocab[feat]] = count

    # create array of testing instance x word count
    testing_array = np.zeros((len(testing_formatted), feat2col))
    for i in range(len(testing_formatted)):
        split = testing_formatted[i].split(" ")
        class_label = split[0]
        word_counts = split[1:-1]

        # keep track of training labels. i-th item in actual_labels is the label for i-th row/vector in array
        testing_labels.append(class_label)

        # change cell word counts
        for pair in word_counts:
            feat = pair.split(":")[0]
            count = int(pair.split(":")[1])

            # ignore OOV
            if feat in vocab:
                testing_array[i, vocab[feat]] = count

    return training_array, training_labels, testing_array, testing_labels


# calculates euclidean distance between two vectors, formatted as np arrays
def euclid(a, b):
    diff = a - b
    dist = sqrt(dot(diff.T, diff))
    return dist


# calculates cosine similarity between two vectors that are formatted as dictionaries
def cosine(a, b):
    dist = dot(a, b) / (norm(a) * norm(b))
    return dist


# measures distance between test vector and all training vectors, returns k neighbors
def neighboring(vec, tr_vecs):
    neighbors = Counter()

    for i in range(len(tr_vecs)):
        if similarity_func == 1:
            distance = euclid(vec, tr_vecs[i, :])

        elif similarity_func == 2:
            distance = cosine(vec, tr_vecs[i])

        else:
            raise Exception("Invalid similarity function requested")

        # we know the class label by the index; the i-th class label should match the row
        neighbors[i] = distance

    # a small euclidean distance indicates that vectors are located in the same region of a vector space
    if similarity_func == 1:
        k_neighbors = neighbors.most_common()[:-k_val - 1:-1]

    # a high cosine similarity indicates that vectors are located in the same general direction from the origin
    else:
        k_neighbors = neighbors.most_common(k_val)

    return k_neighbors


# format data
tr_array, tr_y_real, te_array, te_y_real = vectorizer(training_data, test_data)


# uses majority voting (each of the neighbors has one vote) to classify the vector
def classify(neighbors):
    labels = {"talk.politics.guns": 0.0, "talk.politics.misc": 0.0, "talk.politics.mideast": 0.0}

    # look up the label of the neighboring training instances
    for i, dist in neighbors:
        labels[tr_y_real[i]] += 1

    # divide to get probabilities
    for label in labels:
        labels[label] = (labels[label] / k_val)

    return labels


# track for confusion matrix
tr_y_pred = []
te_y_pred = []

# classify and write to file
with open(sys_output, 'w') as d:
    # TRAINING VECTORS
    d.write("%%%%% training data:\n")

    # get probability distributions
    for i in range(len(tr_array)):
        k_neighbors = neighboring(tr_array[i], tr_array)
        distribution = classify(k_neighbors)
        sorted_dist = sorted(distribution, key=distribution.get, reverse=True)
        tr_y_pred.append(sorted_dist[0])

        d.write("array:" + str(i) + " " + tr_y_real[i] + " ")
        for label in sorted_dist:
            d.write(label + " " + str(distribution[label]) + " ")
        d.write("\n")

    # TEST VECTORS
    d.write("%%%%% test data:\n")

    # get probability distributions
    for i in range(len(te_array)):
        k_neighbors = neighboring(te_array[i], tr_array)
        distribution = classify(k_neighbors)
        sorted_dist = sorted(distribution, key=distribution.get, reverse=True)
        te_y_pred.append(sorted_dist[0])

        d.write("array:" + str(i) + " " + te_y_real[i] + " ")
        for label in sorted_dist:
            d.write(label + " " + str(distribution[label]) + " ")
        d.write("\n")

    # header order for confusion matrix
    label_set = ["talk.politics.guns", "talk.politics.mideast", "talk.politics.misc"]

    # create confusion matrices and accuracy scores
    train_cm = confusion_matrix(tr_y_real, tr_y_pred, labels=label_set)
    train_accuracy = accuracy_score(tr_y_real, tr_y_pred)
    train_formatted = pd.DataFrame(train_cm, index=label_set, columns=label_set)

    test_cm = confusion_matrix(te_y_real, te_y_pred, labels=label_set)
    test_accuracy = accuracy_score(te_y_real, te_y_pred)
    test_formatted = pd.DataFrame(test_cm, index=label_set, columns=label_set)

    pd.set_option('display.expand_frame_repr', False)

    print("Confusion matrix for the training data:")
    print("row is the truth, column is the system output \n")
    print(train_formatted)
    print("\n")
    print("Training accuracy=" + str(train_accuracy))

    print("Confusion matrix for the testing data:")
    print("row is the truth, column is the system output \n")
    print(test_formatted)
    print("\n")
    print("Testing accuracy=" + str(test_accuracy))