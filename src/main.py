from pipeline import process_hard_data
from decisionregressor_traintest import train_and_save as decision_train_save
from decisionregressor_traintest import load_and_test as decision_load_test
from xgboost_traintest import train_and_save as boost_train_save
from xgboost_traintest import load_and_test as boost_load_test
from deepmodel_traintest import train_and_save as deep_train_save
from deepmodel_traintest import load_and_test as deep_load_test
import numpy as np
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(prog='Run', description='Runs the script', epilog='')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--eval', action='store_true')
    args = parser.parse_args()

    if not(args.train ^ args.eval):
        raise ValueError("Train and eval arguments must be distinct")
    elif not(args.train or args.eval):
        raise ValueError("Either --train or --eval must be set as an argument")

    if (args.train):        
        X_train, X_val, y_train, y_val = process_hard_data("", True)                
        decision_train_save(X_train, X_val, y_train, y_val)        
        boost_train_save(X_train, X_val, y_train, y_val)
        deep_train_save(X_train, X_val, y_train, y_val)
        
    elif (args.eval):
        X, y = process_hard_data("test_data.csv", False)
        print("Decision Regressor:")
        decision_load_test(X, y)
        print("")
        print("XGBoost Classifier:")        
        boost_load_test(X, y)
        print("")
        print("PyTorch Deep Model:")
        deep_load_test(X, y)

if __name__ == "__main__":
    main()
