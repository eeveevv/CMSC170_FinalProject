import numpy as np
import csv 

# with open('../dataset/processed_test.csv', 'r') as f:
#     read = csv.reader(f)
#     data = []
#     output = []
#     for x in read:
#         data.append(x[:-1])
#         output.append(x[-1])

#     test_in = np.array(data, dtype=np.float64)
#     test_out = np.array(output, dtype=np.float64)

# with open('../dataset/processed_tune.csv', 'r') as f:
#     read = csv.reader(f)
#     data = []
#     output = []
#     for x in read:
#         data.append(x[:-1])
#         output.append(x[-1])

#     tune_in = np.array(data, dtype=np.float64)
#     tune_out = np.array(output, dtype=np.float64)

# with open('../dataset/processed_train.csv', 'r') as f:
#     read = csv.reader(f)
#     data = []
#     output = []
#     for x in read:
#         data.append(x[:-1])
#         output.append(x[-1])

#     train_in = np.array(data, dtype=np.float64)
#     train_out = np.array(output, dtype=np.float64)

#     learning_rate = 0.0001
# input_size = train_in.shape[1]
# N = train_in.shape[0]
# batch_size = N
# layer_sizes = [16, 16, 16] + [1]

def ReLU(layers):
    if type(layers) == list:
        layers = np.array(layers, dtype=np.float64)
    return np.where(layers < 0, 0, layers)

def ReLUDerivative(layers):
    return np.where(layers < 0, 0, 1)

# weights = [np.zeros((input_size, layer_sizes[0]), dtype=np.float64)]
# biases = []
# layers = [np.zeros((batch_size, input_size), dtype=np.float64)]
# for x in layer_sizes:
#     biases.append(np.ones((x), dtype=np.float64))
#     layers.append(np.zeros((batch_size, x), dtype=np.float64))

# for i in range(0, len(layer_sizes)-1):
#     weights.append(np.zeros((layer_sizes[i], layer_sizes[i+1]), dtype=np.float64))


# print([x.shape for x in layers])
# print([x.shape for x in weights])
# print([x.shape for x in biases])

# iters = 500000
# learning_rate = 0.00001
# for iter in range(iters):

#     # get batch_size datapoints
#     cost = np.zeros((batch_size))
#     indexes = np.random.choice(range(N), size = batch_size, replace= False)
#     layers[0] = train_in[indexes]
#     expected = train_out[indexes].reshape(-1, 1)

#     #forward prop
#     for i in range(len(layers)-1):
#         if i < len(layers)-1: 
#             layers[i+1] = ReLU(layers[i].dot(weights[i])+biases[i])
#         else:
#             layers[i+1] = layers[i].dot(weights[i])+biases[i]

#     cost = (layers[-1] - expected)**2

#     # print(cost)
#     print(f"Mean Squared Error [{iter}]: {cost.mean()}")
#     # backward prop

#     bias_diff = [np.zeros_like(x) for x in biases]
#     weights_diff = [np.zeros_like(x) for x in weights]

#     delta = 2 * (layers[-1] - expected)
#     for i in range(len(layer_sizes) - 1, -1, -1):
#         weights_diff[i] = layers[i].T @ delta / batch_size
#         bias_diff[i] = delta.mean(axis = 0)

#         if i > 0:
#             delta = delta @ weights[i].T * ReLUDerivative(layers[i])

#     for x in range(len(layer_sizes)-1, -1, -1):
#             weights[x] = weights[x] - learning_rate * weights_diff[x]
#             biases[x] = biases[x] - learning_rate * bias_diff[x]

# layers[0] = test_in
# expected = test_out.reshape(-1, 1)

# #forward prop
# for i in range(len(layers)-1):
#     if i < len(layers)-1: 
#         layers[i+1] = ReLU(layers[i].dot(weights[i])+biases[i])
#     else:
#         layers[i+1] = layers[i].dot(weights[i])+biases[i]

# cost = (layers[-1] - expected)**2


# SStotal = (expected - expected.mean())**2
# SStotal = SStotal.sum()

# SSres = (layers[-1] - expected)**2
# SSres = SSres.sum()

# print(f"RMSE [test]: {((layers[-1] - expected)**2).mean() ** 0.5}")
# print(f"MAE [test]: {(layers[-1] - expected).mean()}")
# print(f"MAPE [test]: {np.abs((expected - layers[-1])/expected).mean()*100}%")
# print(f"R2 [test]: {1 - (SSres/SStotal)}")


# # get batch_size datapoints
# layers[0] = tune_in
# expected = tune_out.reshape(-1, 1)

# #forward prop
# for i in range(len(layers)-1):
#     if i < len(layers)-1: 
#         layers[i+1] = ReLU(layers[i].dot(weights[i])+biases[i])
#     else:
#         layers[i+1] = layers[i].dot(weights[i])+biases[i]

# cost = (layers[-1] - expected)**2

# # print(cost)
# SStotal = (expected - expected.mean())**2
# SStotal = SStotal.sum()

# SSres = (layers[-1] - expected)**2
# SSres = SSres.sum()


# print(f"\nRMSE [tune]: {((layers[-1] - expected)**2).mean() ** 0.5}")
# print(f"MAE [tune]: {(layers[-1] - expected).mean()}")
# print(f"MAPE [tune]: {np.abs((expected - layers[-1])/expected).mean()*100}%")
# print(f"R2 [tune]: {1 - (SSres/SStotal)}")

def run(train_x, train_y, test_x, learning_rate=0.00001, iters=500000):
    input_size = train_x.shape[1]
    N = train_x.shape[0]
    batch_size = N
    layer_sizes = [16, 16, 16] + [1]
 
    # layer setup
    weights = [np.zeros((input_size, layer_sizes[0]), dtype=np.float64)]
    biases = []
    layers = [np.zeros((batch_size, input_size), dtype=np.float64)]
    for x in layer_sizes:
        biases.append(np.ones((x), dtype=np.float64))
        layers.append(np.zeros((batch_size, x), dtype=np.float64))
 
    for i in range(0, len(layer_sizes)-1):
        weights.append(np.zeros((layer_sizes[i], layer_sizes[i+1]), dtype=np.float64))
 
    # training
    for iter in range(iters):
        indexes = np.random.choice(range(N), size = batch_size, replace= False)
        layers[0] = train_x[indexes]
        expected = train_y[indexes].reshape(-1, 1)
 
        #forward prop
        for i in range(len(layers)-1):
            if i < len(layers)-1:
                layers[i+1] = ReLU(layers[i].dot(weights[i])+biases[i])
            else:
                layers[i+1] = layers[i].dot(weights[i])+biases[i]
 
        # (the print of the error on every iteration is left out)
 
        # backward prop
        bias_diff = [np.zeros_like(x) for x in biases]
        weights_diff = [np.zeros_like(x) for x in weights]
 
        delta = 2 * (layers[-1] - expected)
        for i in range(len(layer_sizes) - 1, -1, -1):
            weights_diff[i] = layers[i].T @ delta / batch_size
            bias_diff[i] = delta.mean(axis = 0)
 
            if i > 0:
                delta = delta @ weights[i].T * ReLUDerivative(layers[i])
 
        for x in range(len(layer_sizes)-1, -1, -1):
            weights[x] = weights[x] - learning_rate * weights_diff[x]
            biases[x] = biases[x] - learning_rate * bias_diff[x]
 
    # predict the test rows with the trained weights
    layers[0] = test_x
 
    #forward prop
    for i in range(len(layers)-1):
        if i < len(layers)-1:
            layers[i+1] = ReLU(layers[i].dot(weights[i])+biases[i])
        else:
            layers[i+1] = layers[i].dot(weights[i])+biases[i]
 
    return layers[-1].reshape(-1)