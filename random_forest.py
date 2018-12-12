# Importing the required libraries
import numpy as np
import pandas as pd
import sys
import math
import random
from pprint import pprint
from random import seed

from DecisionTree import *

# This file contains the functions needed to implement Random Forest Algorithm.
# ID3 and other functions used by the Random Forest Algorithm are imported from DecisionTree.py
# There are two implementations of Random Forest: Using subsampling of rows and columns.
# test_data_split and sample_rows functions are used for sampling test data and sample data respectively. Instead of these, sample() from the pandas library can also be used.

def test_data_split(data,attributes,ratio):
    
    # The function divides the test data based on random sampling with displacement
    # It takes total data, attributes and the percentage of split as input and returns test data
    
    count = int(round(len(data)* ratio))
    max = len(data) - 1
    test_data = pd.DataFrame(columns = attributes)
    
    while len(test_data) < count:
        
        index = random.randint(0,max)
        test_data = test_data.append(data.loc[index,attributes])
    
    return test_data

def sample_rows(data,attributes,sample_size):
    
    #This function samples training data for each tree based on the sample_size mentioned.
    # Inputs are data, attributes and percentage of data in each sub sample
    # Returns the training data for each tree
    
    train_count = int(round(len(data)* sample_size))
    max = len(data) - 1
    training_data = pd.DataFrame(columns = attributes)
    
    while len(training_data) < train_count:
        
        index = random.randint(0,max)
        training_data = training_data.append(data.loc[index,attributes])
    
    return training_data

def sample_columns(data,input_attributes,target_attr,ratio):
    
    # This function does random sampling of columns and gives sqrt(n) input features for each tree
    # Inputs are data, list of input attributes,target attribute & percent of training data needed
    # Returns a)training data based on random column sampling b) list of columns selected
    
    n_features = int(round(math.sqrt(len(input_attributes))))
    features = list()
    max = len(input_attributes) - 1
    train_count = int(round(len(data)* ratio))
    
    # selecting input attributes by random sampling
    while len(features) <= n_features-1:
        feature = input_attributes[random.randint(0,max)]
        if feature not in features:
            features.append(feature)
    features.append(target_attr)

    # Developing training data for each tree
    training_data = pd.DataFrame(columns=features)
    while len(training_data) < train_count:
        index = random.randint(0,max)
        training_data = training_data.append(data.loc[index,features])

    features.remove(target_attr)
    print features
    return training_data,features


def rf_classify(data,forest,target_attr,index):
    
    # This function predicts the output of test data using the forest provided
    # Inputs are forest, test data and the index of data
    # returns the majority output of the forest
    
    results = list()
    for tree in forest:
        result = classify(tree,data,target_attr,index)
        results.append(result)
    
    return max(set(results),key = results.count)

def rf_accuracy(forest,data,target_attr):
    
    # The function predicts the accuracy of the random forest
    # It takes forest and test data as inputs and gives the accuracy
    
    count = 0
    for index in range(len(data)):
        prediction = rf_classify(data,forest,target_attr, index)
        actual_result = data.iloc[index][target_attr]
        if (prediction == actual_result):
            count += 1
    accuracy = (float(count)/len(data)) *100
    return accuracy


def random_forest_rows(data,attributes,input_attributes,target_attr,num_trees,sample_size,unique_values,default):
    
    # This function uses ID3 decision tree algorithm to create a random forest and calculates the accuracy
    # Inputs are  data, attributes, number of trees in the forest and sample size
    # returns a accuracy of the forest
    
    forest = list()
    for i in range(num_trees):
        training_data = data.sample(frac=sample_size,replace = True, random_state=100)
        tree = ID3(training_data,target_attr,input_attributes,unique_values,default)
        forest.append(tree)

    return forest

def random_forest_columns(data,attributes,input_attributes,target_attr,num_trees,ratio):
    
    # This function uses ID3 decision tree algorithm to create a random forest and calculates the accuracy
    # Inputs are  data, attributes, number of trees in the forest and sample size
    # returns a accuracy of the forest

    forest = list()
    for i in range(num_trees):
        training_data,features = sample_columns(data,input_attributes,target_attr,ratio)
        tree = ID3(training_data,target_attr,features)
        print tree
        forest.append(tree)
    
    return forest

def print_forest(forest):
    # Prints all the trees in the random forest
    for index in range(len(forest)-1):
        pprint.pprint(forest(index))







