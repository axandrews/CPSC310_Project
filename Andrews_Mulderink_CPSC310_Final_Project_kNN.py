# -*- coding: utf-8 -*-
"""
Alexa Andrews and Jeffrey Mulderink
Data Mining CPSC 310 Final Project
KNN

"""


import utils
import numpy as np
import math
import copy



def normalize_attributes(test_set, training_set):
    '''
        A function to normalize training and test data at a given set of data_indices
        The attributes of data_indices should be numeric and continuous. 
        Param training_set: A table of training data
        Param test_set: A table of testing data
        Param data_indices: The indices of the attributes to be normalized        
        Returns: tables containing only the attributes given by data_indices normalized 
                normalized_training_data - normalized training data [0,1]
                normalized_test_data - normalized test data [0,1]
    '''
    normalized_training_data = []
    normalized_test_data = []
    
    for index in range(len(training_set[0])):  # go by column
        training_column = utils.get_column(training_set, index)
        test_column = utils.get_column(test_set, index)
        min_val = min(training_column)
        max_val = max(training_column)
        training_column = normalize(min_val, max_val, training_column)
        test_column = normalize(min_val, max_val, test_column)
        normalized_training_data.append(training_column)
        normalized_test_data.append(test_column)
    
    # we appended columns to normalized data tables, so transpose them    
    normalized_test_data = [utils.convert_to_numeric(list(i)) for i in np.array(normalized_test_data).T]
    normalized_training_data = [utils.convert_to_numeric(list(i)) for i in np.array(normalized_training_data).T]

    return normalized_test_data, normalized_training_data



def make_kNN_prediction(test_instance, training_set, k):
    closest = get_k_closest(test_instance, training_set, k)
    class_labels = utils.get_column(closest, -1)
    return np.median(class_labels)  # the most common of the two class labels will be the median
    
    

def get_k_closest(test_instance, training_data, k=5):
    '''
        A function to get the k closest values to an instance in a set of training data
        Param test_instance: An instance whose nearest neighbors will be found
        Param training_data: A table within which nearest neighbors for the test_instance will be found. Does
        not contain the test instance.
        Returns: A table of the k instances from training_data closest to the test_instance
    '''
    training_data = copy.deepcopy(training_data)
    for i, training_instance in enumerate(training_data):
        training_data[i].append(calculate_distance(test_instance, training_instance))
        
    training_data.sort(key=lambda x:x[-1])

    return training_data[:k]


def normalize(min_val, max_val, column):
    '''
        A simple function to enhance readability for calculating the normalized 
        values for a list given a min and max value
        Param min_val: The minimum value for normalization
        Param max_val: The maximum value for normalization
        Param column: A list to normalize
        Returns: A list containing normalized values for the given info
    '''
    return [max(0, min(1, (x-min_val) / (max_val-min_val))) for x in column]


def calculate_distance(test_instance, training_instance):
    '''
        A simple function to enhance readability for calculating the distance 
        between two instances based on the attributes of interest given by data_indices 
        which contain normalized, numeric values 
        Param test_instance: The first instance
        Param training_instance: The second instance
        Param data_indices: The indices whose values should be used in the distance calculation
        Returns: The distance between the two instances as a float 
    '''
    return math.sqrt(sum([(test_instance[i] - training_instance[i])**2 for i in range(len(test_instance))]))
    
    
def get_stratified_folds(table, k=10):
    '''
        Creates k folds of table which have an equal amount of each class in the column
        at class_index
        Param table: A table to divide into folds
        Param class_index: The index of the attribute of the table which defines the class
        Param categories_dict: An optional dictionary to categorize a continuous attribute at class_index. Default None
        Param k: The number of folds
        Returns: folds, a list of tables which have an equal amount of each class. Latter folds
        may have one fewer per class.
    '''
#    table = copy.deepcopy(table)
#    
    _, groups = utils.group_by(table, len(table[0])-1)
     
    folds = [[] for _ in range(k)]
    for group in groups:
         for i, instance in enumerate(group):
             folds[i % k].append(instance)
    return folds


def create_kNN_classifier(table):
    folds = get_stratified_folds(table)
    
    
    accuracies = {}
    for k in range(3, 100, 6):
        print("testing at k=%d" % k)
        predictions, actuals = [], [] 
        for i, fold in enumerate(folds):
            train = [instance for fold in folds[:i] for instance in fold] + [instance for fold in folds[i+1:] for instance in fold]
            test, train = normalize_attributes(fold, train)
            for test_instance in test:
                predictions.append(make_kNN_prediction(test_instance, train, k))
                actuals.append(test_instance[-1])
        correct = [predictions[i] == actuals[i] for i in range(len(predictions))]
        accuracies[k] = correct.count(True) / len(correct)
    
     
    return accuracies
   
def main():
    header, table = utils.open_csv_with_header("default_of_credit_card_clients.csv")
    
#    header, table = utils.open_csv_with_header("auto-data-no-names.txt")
    np.random.shuffle(table)
    
    accuracies = create_kNN_classifier(table[:200])
    print(accuracies)

main()