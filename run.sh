#!/bin/bash

#Argument options:
# Only 1 argument is accepted. No arguments runs training
# train: Trains all 3 models
# eval: Evaluates all saved models

# Check if the first argument is "eval"
if [ "$1" == "train" ]; then
    echo "Running in training mode..."
    python src/main.py --train
elif [ "$1" == "eval" ]; then
    echo "Running in evaluation mode..."
    python src/main.py --eval
else
    echo "Running in training mode, since no arguments created..."
    python src/main.py --train
fi
