import numpy as np

# Mean Absolute Error (MAE)
def mae(y_true, y_pred):
  return np.mean(np.abs(y_true - y_pred))

# Root Mean Squared Error (RMSE)
def rmse(y_true, y_pred):
  return np.sqrt(np.mean((y_true - y_pred)**2))

# Mean Absolute Percentage Error (MAPE)
def mape(y_true, y_pred):
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# R-squared or Coefficient of Determination (R2)
def r_squared(y_true, y_pred):
  ss_res = np.sum((y_true - y_pred)**2)
  ss_tot = np.sum((y_true - np.mean(y_true))**2)
  return 1 - (ss_res / ss_tot)


#evaluation code metrics from:
#https://medium.com/@lala.ibadullayeva/mastering-regression-metrics-a-comprehensive-guide-for-ml-practitioners-128fea0c28f3
     