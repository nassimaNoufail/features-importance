# import libraries
# requires scikit-learn and xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier

import os
import pandas as pd
import numpy as np
import sys, getopt



def main(argv):
    trainfile = ''
    testfile = ''
    try:
       opts, args = getopt.getopt(argv,"hi:o:",["trainfile=","testfile="])
    except getopt.GetoptError:
       print ('test.py -train <inputfile> -test <outputfile>')
       sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -train <inputfile> -test <outputfile>')
            sys.exit()
        elif opt in ("-train", "--trainfile"):
            trainfile = arg
        elif opt in ("-test", "--testfile"):
            testfile = arg
    print ('Train file is "', trainfile)
    print ('Test file is "', testfile)

    ALGS = ["forest", "extra", "adaboost", "gradientboost", "xgb"]
    ALG_DEFAULT = "xgb"
    PATH_DATASET=r".\dataset"
    FILE_TRAIN = PATH_DATASET + os.sep + trainfile
    FILE_TEST = PATH_DATASET + os.sep + testfile  

    # instantiate variables
    NUMERICAL_FEATURES = []
    CATEGORICAL_FEATURES = []
    CSV_COLUMN_NAMES = []

    # read train and test csv files into a pandas dataframe, return features and labels
    # and create automatically the FIELD_DEFAULTS  to be used to decode csv
    train = pd.read_csv(FILE_TRAIN, sep=";", header=0)
    test = pd.read_csv(FILE_TEST, sep=";", header=0)
    y = train.pop("target")

    # fill NaN in train sets and categorize columns datatype
    for column in train.columns:
        if(train[column].dtype == np.float64 ):
            NUMERICAL_FEATURES.append(column)
            train[column] = train[column].fillna(999)
        elif (train[column].dtype == np.int64):
            NUMERICAL_FEATURES.append(column)
            train[column] = train[column].fillna(990)
        else:
            CATEGORICAL_FEATURES.append(column)
            train[column] = train[column].fillna("UNK")

    # fill NaN in the test set
    for column in test.columns:
        if(test[column].dtype == np.float64 ):
            test[column] = test[column].fillna(999)
        elif (test[column].dtype == np.int64):
            test[column] = test[column].fillna(990)
        else:
            test[column] = test[column].fillna("UNK")

    # convert categorical columns into one hot vectors
    train = pd.get_dummies(train, columns=CATEGORICAL_FEATURES) 
    test = pd.get_dummies(test, columns=CATEGORICAL_FEATURES)

    # build the train dataset
    print ("Starting...")
    features = train.columns

    for ALG_DEFAULT in ALGS:
        if (ALG_DEFAULT == "forest"):
            # create the random forest classifier and fit it to the training set
            classifier = RandomForestClassifier(
                n_estimators=100,
                criterion="gini",
                max_depth=None,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.,
                max_features="auto",
                max_leaf_nodes=None,
                min_impurity_decrease=0.,
                bootstrap=True,
                oob_score=False,
                verbose=4,
                n_jobs=2)
        elif (ALG_DEFAULT == "xgb"):
            classifier = XGBClassifier()
        elif (ALG_DEFAULT == "extra"):
            classifier = ExtraTreeClassifier(
                criterion="gini", 
                splitter="random", 
                max_depth=None, 
                min_samples_split=2, 
                min_samples_leaf=1, 
                min_weight_fraction_leaf=0.0, 
                max_features="auto", 
                random_state=None, 
                max_leaf_nodes=None, 
                min_impurity_decrease=0.0, 
                min_impurity_split=None, 
                class_weight=None)
        elif (ALG_DEFAULT == "adaboost"):
            classifier = AdaBoostClassifier(
                n_estimators=100
            )
        elif (ALG_DEFAULT == "gradientboost"):
            classifier = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=1.0,
                max_depth=1, 
                random_state=0)
        else:
            classifier = ""
        print (ALG_DEFAULT) # print the currently used classifier
        
        # fit our classifier to our train dataset
        classifier.fit(train[features], y)

        # make predictions ovet the test dataset and evaluate
        preds = classifier.predict(test[features])
        pd.crosstab(test['target'], preds, rownames=['actual'], colnames=['preds'])

        # display the relevance of each feature in a sorted way (from most relevant to not relevant)
        result = list(zip(train[features], classifier.feature_importances_))
        result = sorted(result, key=lambda result: result[1], reverse=True)
        i = 0
        for important_column in result:
            if (i<10):
                print(important_column)
                i = i + 1

if __name__ == "__main__":
   main(sys.argv[1:])