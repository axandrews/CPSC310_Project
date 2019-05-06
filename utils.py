# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:10:15 2019

@author: alexaandrews16
"""

# utils.py 
# file containing re-usable, general utility variables and functions

import csv
import numpy as np
import copy
import math


def open_csv_with_header(infile):
    data = list(csv.reader(open(infile)))
    header = data[0]
    data = data[1:]
    for i, r in enumerate(data):
        convert_to_numeric(data[i])
    return header, data


def get_frequencies(table, col_index):
    '''
        returns parallel array
    '''
    values = []
    counts = []
    column = sorted(get_column(table, col_index))
    
    for value in column:
        if value not in values:  # first time we see it
            values.append(value)
            counts.append(1)
        else:  # it is sorted so we just saw it
            counts[-1] += 1
    return values, counts


def get_occurrences(table, col_index):
    '''
        Return dictionary
    '''
    occurrences = {}
    
    column = get_column(table, col_index)   
    for entry in column:
        if entry in occurrences:
            occurrences[entry] += 1
        else:
            occurrences[entry] = 1
    return occurrences


def get_column(table, col_index):
    '''
        Gets a column from a table
        Param table: a 2d array
        Param col_index: the desired column from the 2d array
        Returns: A 1d array representing the column index by col_index in table
    '''
    column = []
    for row in table:
        if row[col_index] != "NA" :
            column.append(row[col_index])
    return column


def group_by(table, column_index):
    '''
        Partitions a table by an attribute, returning a tuple of parallel lists, the first containing
        the attribute value of each group, the second containing the groups as a list of tables
    '''
    group_names = sorted(list(set(get_column(table, column_index))))
    
    # now we need a list of subtables each corresponding to a value in group_names
    groups = [[] for name in group_names]
    for row in table:
        # which group does it belong to?
        group_by_value = row[column_index]
        if group_by_value is "NA":
            continue
        index = group_names.index(group_by_value)
        groups[index].append(row)
        
    return group_names, groups


def write_table(file_name, table):
    '''
        writes a table to an output file in csv form
        Param file_name: The desired name of the output file
        Param table: The table to write to file
        Returns: None
    '''
    outfile = open(file_name, "w")
    output = ""
    for row in table:
        for col in range(0, len(row)):
            output += str(row[col])
            if col < len(row) - 1:
                output += (",")
        output += "\n"
    outfile.write(output)
    outfile.close()    
    

def read_table(file_name):
    '''
        reads a csv file into a 2d array "table"
        Param file_name: the name of the input fle
        Returns: a table representing the contents of input file
                any numeric values are converted to their appropriate type
    '''
    table = []  # will be a nested list
    # open the file
    # "r" is read mode "w" is write mode "a" is append mode
    infile = open(file_name, "r")  
    # read each line in infile and append it to table as a row (1d list)
    # we could use a library like the csv module but we will do by hand here
    lines = infile.readlines()
    # print(lines)
    for line in lines:
        # get ride of new line
        line = line.strip()  # strips whitespace characters, inc newlines
        # now we want to break line into individual strings using comma as a
        # delimiter. Split splits a string based on a delimiter.
        values = line.split(",")
        # print(values)  # values is a 1d array
        convert_to_numeric(values)
        table.append(values)  # add it to the end of table
        
    infile.close()
    return table
    

# if possible converts strings in array values to int
def convert_to_numeric(values):
    ''' 
        converts values in an array to int or float values where possible
        Param values: an array whose values will be converted to numeric
        Returns: None, directly modifies parameter values
    '''
    for i in range (len(values)):
        try:
            numeric_val = int(values[i])
            values[i] = numeric_val
        except ValueError:
            try:
                numeric_val = float(values[i])
                values[i] = numeric_val
            except ValueError:
                pass
    return values

def print_instance(instance):
    '''
        Prints a list as a row of commma separated values 
    '''
    for i, item in enumerate(instance):
        if i < len(instance) - 1:
            print(item, end=', ')
        else:
            print(item)
            
            
            
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
        training_column = get_column(training_set, index)
        test_column = get_column(test_set, index)
        min_val = min(training_column)
        max_val = max(training_column)
        training_column = normalize(min_val, max_val, training_column)
        test_column = normalize(min_val, max_val, test_column)
        normalized_training_data.append(training_column)
        normalized_test_data.append(test_column)
    
    # we appended columns to normalized data tables, so transpose them    
    normalized_test_data = [convert_to_numeric(list(i)) for i in np.array(normalized_test_data).T]
    normalized_training_data = [convert_to_numeric(list(i)) for i in np.array(normalized_training_data).T]

    return normalized_test_data, normalized_training_data


def make_kNN_prediction(test_instance, training_set, k):
    closest = get_k_closest(test_instance, training_set, k)
    class_labels = get_column(closest, -1)
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
    _, groups = group_by(table, len(table[0])-1)
     
    folds = [[] for _ in range(k)]
    for group in groups:
         for i, instance in enumerate(group):
             folds[i % k].append(instance)
    return folds