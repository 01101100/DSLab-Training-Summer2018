import numpy as np

def get_data(path):
    with open(path) as file:
        raw_data = file.read().splitlines()

    X, Y = [],[]
    for line in raw_data:
        line_array = line.split()
        X.append(line_array[1:len(line_array) - 1])
        Y.append(float(line_array[-1]))

    # print(np.matrix(X))
    # print(Y)
    return X, np.array(Y).reshape(-1,1)

def normalize_and_add_ones(X):
    X = np.array(X).astype(float)
    # X_max = np.array([[np.amax(X[:, column_id]) for column_id in range(X.shape[1])] for _ in range(X.shape[0])])
    X_max = np.tile(np.max(X, axis=0), (X.shape[0], 1))

    # X_min = np.array([[np.amin(X[:, column_id])
    #                    for  column_id in range(X.shape[1])]
    #                   for _ in range(X.shape[0])])
    X_min = np.tile(np.min(X, axis=0), (X.shape[0], 1))
    X_normalized = (X - X_min) / (X_max - X_min)

    ones = np.array([[1] for _ in range(X_normalized.shape[0])])

    return np.column_stack((ones, X_normalized))
