# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:10:15 2019

@author: alexaandrews16
"""

# utils.py 
# file containing re-usable, general utility variables and functions

import csv


header = ["Car Name", "Model Year", "MSRP"]
msrp_table = [["ford pinto", 75, 2769], ["toyota corolla", "NA", 2711],
              ["ford pinto", 76, 3025], ["toyota corolla", 77, 2789],
              ["toyota corolla", 75, 2749], ["ford mustang", 76, 3322],
              ["vw rabbit",75, 3499], ["chevrolet impala", 71, 3811]]


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