import csv
import math
 
K_VALUE = 7
P_VALUE = 2
 
TRAIN_PATH = "../dataset/processed_train.csv"
TUNE_PATH = "../dataset/processed_tune.csv"
TEST_PATH = "../dataset/processed_test.csv"
 
 
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
        next(csv_reader)  # skip the header row
 
        for row in csv_reader:
            point = []
            for value in row:
                point.append(float(value))
            dataset.append(point)
 
    return dataset
 
 
def group_dataset(dataset):
    features = []
    prices = []
 
    for line in dataset:
        features.append(line[0:6]) # the first 6 columns
        prices.append(line[6]) # price (last column)
 
    return features, prices
 

# makes the features comparable 
# for each feature, their means and standard deviation is computed
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
 
 
def compute_closest_matches(test_x, train_x, train_y):
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
        closest_matches.append(distances[:K_VALUE])
 
    return closest_matches
 
 
def predict(closest_matches):
    # average the price of the K closest train cases
    predictions = []
 
    for neighbors in closest_matches:
        total = 0
        for neighbor in neighbors:
            total += neighbor[1]
        predictions.append(total / K_VALUE)
 
    return predictions
 
