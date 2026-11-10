# comparison of all methods, with tables and graphs
# runs baseline, ridge, knn and MLP on the same data and compares them

# notes:
# make mlp.py & ridge.py para mas madali ma-test
# add a run() in each method
#   def run(train_x, train_y, test_x):

import csv
import numpy as np
import matplotlib.pyplot as plt
import EvaluationMetrics as em

# needed directories
input_directory = "../dataset"
output_directory = "../docs/assets"

# baseline
def run_baseline(train_x, train_y, test_x):
    return test_x[:, 2] + test_x[:, 3]

# methods
# (baseline, ridge, knn, MLP)

# use EvaluationMetrics.py
def compute(y_true, y_pred):
    return {
        "MAE": em.mae(y_true, y_pred),
        "RMSE": em.rmse(y_true, y_pred),
        "MAPE": em.mape(y_true, y_pred),
        "R2": em.r_squared(y_true, y_pred),
    }