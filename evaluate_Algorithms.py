#importing the required libraries and all funtions from decision tree.py and random_forest.py
import numpy as np
import pandas as pd
from DecisionTree import *
from random_forest import *

def main():
    
    # Reads input file, creates input and target attributes
    data=pd.read_csv(sys.argv[1], sep = ',')
    attributes = list(data.columns.values)

    input_attributes = attributes[:-1]
    print "\nInput attributes for the dataset:"
    print input_attributes

    target_attr = attributes[-1]
    print "\nTarget attribute for the dataset is:"
    print target_attr

    # finding unique values in input attributes
    unique_values = {}
    for column in input_attributes:
        unique_values[column] = data.loc[:,column].unique()
    
    # defualt gives the majority value in the target attribute and is passed to ID3 function
    default = majority(data,target_attr)

    #creating training and test data
    training_data,test_data = datasplit(data,0.8)

    tree = ID3(training_data,target_attr,input_attributes, unique_values, default)
    print "\nDecision Tree created"

    training_accuracy =  accuracy(tree,training_data,target_attr)
    print "\nAccuracy of training data using ID3:"
    print training_accuracy

    test_accuracy = accuracy(tree,test_data,target_attr)
    print "\nAccuracy of test data using ID3:"
    print test_accuracy

    print "\n Starting random forest execution"
        
    forest = random_forest_rows(data,attributes,input_attributes,target_attr,10,0.2,unique_values,default)

    training_accuracy =  rf_accuracy(forest,training_data,target_attr)
    print "\nAccuracy of training data using Random Forest is:"
    print training_accuracy
    
    test_accuracy = rf_accuracy(forest,test_data,target_attr)
    print "\nAccuracy of test data using Random Forest:"
    print test_accuracy


main()



