import numpy as np
import csv 
# import EvaluationMetrics as em

# with open('../dataset/processed_test.csv', 'r') as f:
#     read = csv.reader(f)
#     data = []
#     output = []
#     for x in read:
#         data.append(x[:-1])
#         output.append(x[-1])

#     test_in = np.array(data)
#     test_out = np.array(output)

# with open('../dataset/processed_tune.csv', 'r') as f:
#     read = csv.reader(f)
#     data = []
#     output = []
#     for x in read:
#         data.append(x[:-1])
#         output.append(x[-1])

#     tune_in = np.array(data)
#     tune_out = np.array(output)

# with open('../dataset/processed_train.csv', 'r') as f:
#     read = csv.reader(f)
#     data = []
#     output = []
#     for x in read:
#         data.append(x[:-1])
#         output.append(x[-1])

#     train_in = np.array(data)
#     train_out = np.array(output)

# train_in

# LAMB = 1  # hyperparameter, the lambda value

def compute_train_stats(X, y):
    x_mean = np.mean(X, axis=0) 			# one mean per column
    x_std = np.std(X, axis=0)				# one std per column
    y_mean = np.mean(y)
    return x_mean, x_std, y_mean

def standardize(X, x_mean, x_std):
    Xs = (X - x_mean) / x_std
    return Xs

def solve_coefficients(lamb, Xs, yc):
    p = Xs.shape[1]
    A = Xs.T @ Xs + lamb * np.eye(p)	  	# XTX + lambI
    b = Xs.T @ yc  							# XTy
    coeff = np.linalg.solve(A, b)
    return coeff

def fit(lamb, X, y):
    # get the training stats
    x_mean, x_std, y_mean = compute_train_stats(X, y)

	# standardize X and center y
    Xs = standardize(X, x_mean, x_std)
    yc = y - y_mean

	# solve for coeffs
    coeff = solve_coefficients(lamb, Xs, yc)
    intercept = y_mean

    return {
        'coeff': coeff,
        'intercept': intercept,
        'x_mean': x_mean,
        'x_std': x_std,
    }

def predict(model, X):
    # standardize X using the training statistics
    Xs = standardize(X, model['x_mean'], model['x_std'])

    # apply the intercept and coefficients
    y_pred = model['intercept'] + Xs @ model['coeff']
    return y_pred

# # convert CSV strings to numbers
# X_train = train_in.astype(float)
# y_train = train_out.astype(float)
# X_tune = tune_in.astype(float)
# y_tune = tune_out.astype(float)

# # train, the predict
# model = fit(LAMB, X_train, y_train)
# y_pred = predict(model, X_tune)

# provinces = ['Metropolitan Manila', 'Nueva Ecija', 'Iloilo', 'South Cotabato', 'Basilan', 'Maguindanao', 'Ifugao', 'Kalinga', 'Ilocos Norte', 'Ilocos Sur', 'Cagayan', 'Isabela',
#  'Quirino', 'Bataan', 'Bulacan', 'Batangas', 'Cavite', 'Laguna', 'Quezon', 'Rizal', 'Marinduque', 'Occidental Mindoro', 'Oriental Mindoro', 'Palawan', 'Romblon', 'Zamboanga del Sur',
#  'Albay', 'Catanduanes', 'Masbate', 'Sorsogon', 'Aklan', 'Guimaras', 'Negros Occidental', 'Bohol', 'Siquijor', 'Biliran', 'Leyte', 'Northern Samar', 'Samar', 'Southern Leyte', 'Bukidnon',
#  'Camiguin', 'Lanao del Norte', 'Misamis Occidental', 'Misamis Oriental', 'Davao Oriental', 'Davao del Norte', 'North Cotabato', 'Sarangani', 'Agusan del Norte', 'Surigao del Norte',
#  'Surigao del Sur']

# labels = []
# with open('../dataset/wfp_food_prices_phl.csv', 'r') as f:
#     read = csv.reader(f)
#     for x in read:
#         if x[0][:4] == '2025' and x[8] == 'Rice (regular, milled)' and x[12] == 'Retail' and x[2] in provinces:
#             labels.append({
#                 'year' : int(x[0][:4]),
#                 'month' : int(x[0][5:7]),
#                 'province' : x[2],
#                 'price' : float(x[14]),
#             })

# # Stop if the labels do not line up with the 2025 rows
# assert len(labels) == len(y_tune)
# for i in range(len(labels)):
#     assert labels[i]['month'] == X_tune[i, 0]
#     assert labels[i]['price'] == y_tune[i]

# # Print each prediction next to the correct answer
# print(f"{'Month':>5}  {'Province':<22} {'Predicted':>10} {'Actual':>10} {'Error':>8}")
# for label, pred, actual in zip(labels, y_pred, y_tune):
#     print(f"{label['month']:>5}  {label['province']:<22} {pred:>10.2f} {actual:>10.2f} {pred - actual:>8.2f}")

# print(f"MAE:  	₱{em.mae(y_tune, y_pred):.4f}/kg")
# print(f"RMSE: 	₱{em.rmse(y_tune, y_pred):.4f}/kg")
# print(f"MAPE: 	{em.mape(y_tune, y_pred):.4f}%")
# print(f"R2:   	{em.r_squared(y_tune, y_pred):.4f}")

# used by Comparison.py
def run(train_x, train_y, test_x, lamb=1):
    model = fit(lamb, train_x, train_y)
    return predict(model, test_x)
