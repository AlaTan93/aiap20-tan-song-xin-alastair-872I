from hard_pipeline import process_hard_data
from soft_pipeline import process_soft_data
from xgboost_traintest import train_and_save as boost_train_save
from decisionregressor_traintest import train_and_save as decision_train_save
from decisionregressor_traintest2 import train_and_save_soft as decision_train_save_soft
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
        X_soft_train, X_soft_val, y_soft_train, y_soft_val = process_soft_data("", True)
        X_train, X_val, y_train, y_val = process_hard_data("", True)        
        decision_train_save(X_soft_train, X_soft_val, y_soft_train, y_soft_val)
        #decision_train_save(X_train, X_val, y_train, y_val)
        
        #boost_train_save(X_train, X_val, y_train, y_val)
        
    elif (args.eval):
        pass

if __name__ == "__main__":
    main()
