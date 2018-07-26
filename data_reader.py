import numpy as np

def get_data(path):
    X, Y = 1,2
    return X, Y

def normalize_and_add_ones(X):
    X = np.array(X)
    X_max = np.array([[np.amax(X[:, column_id])
                       for column_id in range(X.shape[1])]
                      for _ in range(X.shape[0])])

    X_min = np.array([[np.amin(X[:, column_id])
                       for  column_id in range(X.shape[1])]
                      for _ in range(X.shape[0])])

    X_normalized = (X - X_min) / (X_max - X_min)

    ones = np.array([[1] for _ in range(X_normalized.shape[0])])

    return np.column_stack((ones, X_normalized))