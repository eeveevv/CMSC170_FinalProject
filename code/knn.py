import csv
import math
import numpy as np

import EvaluationMetrics as em

P_VALUE = 1
 
TRAIN_PATH = "../dataset/processed_train.csv"
TUNE_PATH = "../dataset/processed_tune.csv"

WFP_PATH = "../dataset/wfp_food_prices_phl.csv" 

PROVINCES = ['Metropolitan Manila', 'Nueva Ecija', 'Iloilo', 'South Cotabato', 'Basilan', 'Maguindanao', 'Ifugao', 'Kalinga', 'Ilocos Norte', 'Ilocos Sur', 'Cagayan', 'Isabela',
 'Quirino', 'Bataan', 'Bulacan', 'Batangas', 'Cavite', 'Laguna', 'Quezon', 'Rizal', 'Marinduque', 'Occidental Mindoro', 'Oriental Mindoro', 'Palawan', 'Romblon', 'Zamboanga del Sur',
 'Albay', 'Catanduanes', 'Masbate', 'Sorsogon', 'Aklan', 'Guimaras', 'Negros Occidental', 'Bohol', 'Siquijor', 'Biliran', 'Leyte', 'Northern Samar', 'Samar', 'Southern Leyte', 'Bukidnon',
 'Camiguin', 'Lanao del Norte', 'Misamis Occidental', 'Misamis Oriental', 'Davao Oriental', 'Davao del Norte', 'North Cotabato', 'Sarangani', 'Agusan del Norte', 'Surigao del Norte',
 'Surigao del Sur']
 
def reader(path):
    # columns
        # month
        # national average change percent = change in the average price from two months ago to last month
        # last month price gap = specific to each province, how much far (in peso) the price of the rice is from the national average
        # last month = national average price last month
        # 2 month = national average price 2 months ago
        # 3 month = national average price 3 months ago

    dataset = []
    with open(path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file)
 
        for row in csv_reader:
            point = []
            for value in row:
                point.append(float(value))
            dataset.append(point)
 
    return dataset

# [1]A. Kumar, “KNN Algorithm: When? Why? How? - TDS Archive - Medium,” 
# Medium, May 25, 2020. https://medium.com/data-science/knn-algorithm-what-
# when-why-how-41405c16c36f (accessed Nov. 10, 2026).

def compute_K(train_x):
    return(round(math.sqrt(len(train_x))))

def group_dataset(dataset):
    features = []
    prices = []
 
    for line in dataset:
        features.append(line[0:6]) # the first 6 columns
        prices.append(line[6]) # price (last column)
 
    return features, prices
 

# makes the features comparable 
# for each feature, their means and standard deviation is computed
# [2]W. H. Khoong, “Why Scaling Your Data Is Important,” CodeX, Jan. 21, 2023. 
# https://medium.com/codex/why-scaling-your-data-is-important-1aff95ca97a2 
# (accessed Nov. 10, 2026).
def compute_scaling(train_features):
    means = []
    stds = []

    for col in range(6):
        total = 0
        for row in train_features:
            total += row[col]
        mean = total / len(train_features)
 
        var = 0
        for row in train_features:
            var += (row[col] - mean) ** 2
        std = math.sqrt(var / len(train_features))
 
        means.append(mean)
        stds.append(std)
 
    return means, stds
 
# scaling technique: standardization
def scale(features, means, stds):
    scaled = []
 
    for row in features:
        new_row = []
        for col in range(6):
            new_row.append((row[col] - means[col]) / stds[col]) #z-score formula
        scaled.append(new_row)
 
    return scaled

# reference: 
# https://medium.com/@buyaalfariz/the-fundamentals-of-k-nearest-neighbors-normalization-and-standardization-a3e6ca616d57
 
 
def get_distance(item):
    return item[0]
 
 
def compute_closest_matches(test_x, train_x, train_y, k):
    closest_matches = []
 
    # loop through each test case
    for i in range(len(test_x)):
        distances = []
 
        # loop through each train case
        for j in range(len(train_x)):
 
            # Minkowski distance (Euclidean)
            total = 0
            for col in range(6):
                total += abs(test_x[i][col] - train_x[j][col]) ** P_VALUE
            distance = total ** (1 / P_VALUE)
 
            distances.append((distance, train_y[j]))
 
        # sort the instances, then keep the K closest
        distances.sort(key=get_distance)
        closest_matches.append(distances[:k])
 
    return closest_matches
 
 
def predict(closest_matches, k):
    # average the price of the K closest train cases
    predictions = []
 
    for neighbors in closest_matches:
        total = 0
        for neighbor in neighbors:
            total += neighbor[1]
        predictions.append(total / k)
 
    return predictions

def run(train_x, train_y, test_x):
    train_x = np.asarray(train_x, dtype=float).tolist()
    train_y = np.asarray(train_y, dtype=float).tolist()
    tune_x = np.asarray(train_x, dtype=float).tolist()

    k = compute_K(train_x)

    means, stds = compute_scaling(train_x)
    train_x = scale(train_x, means, stds)
    test_x = scale(test_x, means, stds)

    closest = compute_closest_matches(test_x, train_x, train_y, k)
    return np.array(predict(closest,k))

def main():
    train = reader(TRAIN_PATH)
    tune = reader(TUNE_PATH)

    train_x, train_y = group_dataset(train)
    tune_x, tune_y = group_dataset(tune)

    tune_prediction = run(train_x, train_y, tune_x)

    actual = np.asarray(tune_y, dtype=float)
    predicted = np.asarray(tune_prediction, dtype=float)
    k = compute_K(train_x)

    means, stds = compute_scaling(train_x)
    train_x = scale(train_x, means, stds)
    tune_x = scale(tune_x, means, stds)

    closest = compute_closest_matches(tune_x, train_x, train_y, k)
    tune_pred = predict(closest, k)

    # EvaluationMetrics subtracts the arrays directly, so lists are converted first
    actual = np.asarray(tune_y, dtype=float)
    predicted = np.asarray(tune_pred, dtype=float)

    print(f"K = {k}, P = {P_VALUE}")
    print(f"tune | MAE: {em.mae(actual, predicted):.3f}  RMSE: {em.rmse(actual, predicted):.3f}  R2: {em.r_squared(actual, predicted):.3f}  MAPE: {em.mape(actual, predicted):.2f}%")

if __name__ == "__main__":
    main()