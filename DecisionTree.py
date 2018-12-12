#Importing the required libraries
import numpy as np
from pprint import pprint
import pandas as pd
import sys
import math
import random
from random import seed


def datasplit(data,split_size):

    #splits the data into training and test data based on random sampling. The random_state attribute in the sample function is used for seed value
    #The function takes data and the percentage split as inputs and returns the training and test data

    training_data = data.sample(frac=split_size, random_state = 100)
    test_data = data.drop(training_data.index)

    return training_data,test_data

def entropy(data,target_attr):
    
    #This function calculates the entropy of the data provided.
    # Takes data and the target attribute as inputs and returns the entopy value
    
    entropy = 0
    probability = data.groupby(target_attr).size().reset_index(name='count')
    
    counts = probability.iloc[:,1].values.astype('float')/len(data)
    for value in counts:
        if value != 0.0:
            entropy -= value * np.log2(value)

    return entropy

def information_gain(data,attribute, target_attr):
    
    #This function calcualtes the information gain for the attribute mentioned, using entropy
    #Inputs are data, target attribute and the attribute for which information gain has to be calculated.
    #returns the information gain
    
    entropy_attr = 0.0

    x = data.loc[:,attribute]
    for value in x.unique():
        data_subset = data.loc[data[attribute] == value]
        entropy_attr += ((len(data_subset)/float(len(data))) * entropy(data_subset,target_attr))

    gain = entropy(data,target_attr) - entropy_attr
    return gain

def choose_node(data,input_attributes, target_attr):
    
    #This function calculates the information gain for each input attribute in the given data and picks the best attribute
    #Takes data, input attributes and target attribute as inputs
    # returns the attribute with the maximum information gain
    
    max_gain = 0.0
    for value in input_attributes:
        attr_gain = information_gain(data, value, target_attr)
        if attr_gain >= max_gain:
            max_gain = attr_gain
            best_attr = value
    return best_attr


def majority(data,attribute):
    #returns the majority value in the attribute passed
    return data[attribute].value_counts().idxmax()


def ID3(data,target_attr, input_attributes,unique_values,default):
    
    #ID3 function generates the decision tree by using information gain
    #It takes the data, input and target attributes as arguments and returns the decision tree
    
    x = data.loc[:,target_attr]
    
    if len(x.unique()) == 1:
        return data.iloc[0][target_attr]
    elif len(input_attributes) == 0:
        return majority(data,target_attr)
    else:
        best_attr = choose_node(data,input_attributes, target_attr)
        tree = {best_attr:{}}
        input_attributes = [x for x in input_attributes if (x != best_attr)]
        
        # Building a subtree for each unique value in the best attribute

        for value in unique_values[best_attr]:
            data_subset = data.loc[data[best_attr] == value]
            if len(data_subset) == 0:
                tree[best_attr][value] = default
            else:
                sub_tree = ID3(data_subset,target_attr,input_attributes,unique_values,default)
                tree[best_attr][value] = sub_tree
    return tree

def classify(tree,test_data,target_attr, index):
    
    #The function classifies each of the test data instance into the values of target attribute
    #It takes test data, decision tree generated, target attribute and the index of the test data as inputs.
    # returns the classified output
    
    if isinstance(tree,basestring):
        #checks if root is the output value
        return tree
    else:
        root = tree.keys()
        sub_tree = tree[root[0]]
        value = test_data.iloc[index][root[0]]
        
        if isinstance(sub_tree[value], basestring):
            return sub_tree[value]
        else:
            return classify(sub_tree[value],test_data,target_attr,index)


def accuracy(tree,data,target_attr):
    
    # This function compares the actual result in the test data with the predicted value and computes the accuracy
    # Takes the decision tree built, data and target attributes as inputs
    # returns the accuracy
    
    count = 0
    for index in range(len(data)):
        result = (classify(tree,data,target_attr,index))
        actual_result = data.iloc[index][target_attr]
        if (result == actual_result):
            count += 1
    accuracy = (float(count)/len(data)) *100
    return accuracy












