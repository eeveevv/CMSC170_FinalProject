# comparison of all methods, with tables and graphs
# runs baseline, ridge, knn and MLP

# notes:
# make mlp.py & ridge.py para mas madali ma-test
# add a run() in each method
#   def run(train_x, train_y, test_x):

import sys
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import EvaluationMetrics as em
# import methods
import RidgeRegression
import knn
import MultilevelPerceptron

# needed directories
input_directory = "../dataset/"
output_directory = "../docs/assets/"

# hyperparameters of each method (update after tuning)
ridge_lambda = 1
knn_k = 7
knn_p = 2
learning_rate = 0.01
iterations = 3000

#reads one processed csv file
#the last col is the target price, the other col are inputs
def read_data(name):
    inputs = []
    targets = []
 
    with open(input_directory + name, "r") as file:
        reader = csv.reader(file)
        #split row into inputs and target
        for row in reader:
            inputs.append(row[:-1])
            targets.append(row[-1])

    # convert csv content to float 
    inputs = np.array(inputs, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    return inputs, targets

# reads the training, validation, and test files
def load_data():
    train_x, train_y = read_data("processed_train.csv")
    tune_x, tune_y = read_data("processed_tune.csv")
    test_x, test_y = read_data("processed_test.csv")
 
    return train_x, train_y, tune_x, tune_y, test_x, test_y

# baseline
# assume next month's price = last month's price
# last month's price = price gap (col 2) + national average 1 month back 
def run_baseline(train_x, train_y, test_x):
     return test_x[:, 2] + test_x[:, 3]

# compute each evaluation, use existing evaluation metrics
def compute(y_true, y_pred):
    mae = em.mae(y_true, y_pred)
    rmse = em.rmse(y_true, y_pred)
    mape = em.mape(y_true, y_pred)
    r2 = em.r_squared(y_true, y_pred)
 
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }

# now run the methods
# saves the scores, time, and predictions of one method
def add_result(results, name, predictions, elapsed, eval_y):
    # make the predictions into one flat list
    predictions = np.array(predictions).reshape(-1)
    results[name] = compute(eval_y, predictions)
    results[name]["Time"] = elapsed
    results[name]["pred"] = predictions

# runs every method
def run_all(train_x, train_y, eval_x, eval_y):
    results = {}
    # baseline
    start = time.time()
    predictions = run_baseline(train_x, train_y, eval_x)
    elapsed = time.time() - start
    add_result(results, "Baseline", predictions, elapsed, eval_y)
 
    # ridge regression
    start = time.time()
    predictions = RidgeRegression.run(train_x, train_y, eval_x, lamb=ridge_lambda)
    elapsed = time.time() - start
    add_result(results, "Ridge", predictions, elapsed, eval_y)
 
    # knn
    start = time.time()
    predictions = knn.run(train_x, train_y, eval_x, k=knn_k, p=knn_p)
    elapsed = time.time() - start
    add_result(results, "KNN", predictions, elapsed, eval_y)
 
    # MLP
    start = time.time()
    predictions = MultilevelPerceptron.run(train_x, train_y, eval_x, learning_rate=learning_rate, iters=iterations)
    elapsed = time.time() - start
    add_result(results, "MLP", predictions, elapsed, eval_y)
 
    return results

# display results
columns = ["MAE", "RMSE", "MAPE", "R2", "Time"]

# prints the table in the terminal
def print_results(results):
    # header
    print("{:<12}".format("Method"), end="")
    for column in columns:
        print("{:>12}".format(column), end="")
 
    print()
    # one row per method
    for name in results:
        print("{:<12}".format(name), end="")
 
        for column in columns:
            print("{:>12.4f}".format(results[name][column]), end="")
 
        print()

# saves the table as csv
def save_csv(results, path):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
 
        writer.writerow(["Method"] + columns)
        for name in results:
            row = [name]
 
            for column in columns:
                row.append(round(results[name][column], 4))
            writer.writerow(row)

# saves the table as html for project page to load
def save_html(results, path):
    html = "<table>\n"
    html += "<tr><th>Method</th>"

    for column in columns:
        html += "<th>" + column + "</th>"

    html += "</tr>\n"
    for name in results:
        html += "<tr><td>" + name + "</td>"

        for column in columns:
            value = results[name][column]
            html += "<td>" + format(value, ".4f") + "</td>"

        html += "</tr>\n"

    html += "</table>"
    file = open(path, "w")
    file.write(html)
    file.close()

# graph 1: mae and rmse of each method
def plot_error (results, path, split):
    names = list(results.keys())
 
    # get the mae and rmse of every method
    mae = []
    rmse = []
 
    for name in names:
        mae.append(results[name]["MAE"])
        rmse.append(results[name]["RMSE"])
 
    x = np.arange(len(names))
 
    plt.figure(figsize=(7, 4))
 
    # two bars per method
    plt.bar(x - 0.2, mae, 0.4, label="MAE")
    plt.bar(x + 0.2, rmse, 0.4, label="RMSE")
 
    plt.xticks(x, names)
    plt.ylabel("Error (PHP/kg)")
    plt.title("MAE and RMSE per method (" + split + ")")
    plt.legend()
 
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

# graph 2: actual vs predicted price, one plot per method
# the red line is a perfect prediction
def plot_pred_vs_actual(results, eval_y, path, split):
    names = list(results.keys())
 
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
 
    # the red line goes from the lowest to the highest actual price
    low = eval_y.min()
    high = eval_y.max()
 
    for i in range(len(names)):
        name = names[i]
 
        # each dot is one province in one month
        axes[i].scatter(eval_y, results[name]["pred"], s=6, alpha=0.5)
        axes[i].plot([low, high], [low, high], "r--", linewidth=1)
 
        axes[i].set_title(name)
        axes[i].set_xlabel("Actual (PHP/kg)")
        axes[i].set_ylabel("Predicted (PHP/kg)")
 
    fig.suptitle("Predicted vs Actual Price (" + split + ")")
 
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    # test only if we type test
    #it validates by default since mas marami need ivalidate
    split = "val"
 
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        split = "test"
 
    train_x, train_y, tune_x, tune_y, test_x, test_y = load_data()

    if split == "test":
        eval_x = test_x
        eval_y = test_y
 
        print("Scoring on the TEST set (2026).")
    else:
        eval_x = tune_x
        eval_y = tune_y
 
        print("Scoring on the VALIDATION set (2025).")
 
    # run all methods and print the table
    results = run_all(train_x, train_y, eval_x, eval_y)
 
    print_results(results)
    # save results
    save_csv(results, output_directory + "results_" + split + ".csv")
    save_html(results, output_directory + "results_" + split + ".html")
 
    #save graphs
    plot_error(results, output_directory + "error_bars_" + split + ".png", split)
    plot_pred_vs_actual(results, eval_y, output_directory + "pred_vs_actual_" + split + ".png", split)
 
    print("Saved tables and graphs to " + output_directory)
 

main()