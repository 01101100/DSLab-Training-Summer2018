from RidgeRegression import RidgeRegression
import data_reader
import numpy as np

if __name__ == "__main__":
    X, Y = data_reader.get_data(path="x28.txt")

    # normalization
    X = data_reader.normalize_and_add_ones(X)
    X_train, Y_train = X[:50], Y[:50]
    X_test, Y_test = X[50:], Y[50:]
    ridge_regression = RidgeRegression()
    best_LAMBDA = ridge_regression.get_the_best_LAMBDA(X_train, np.array(Y_train))
    print("best LAMBDA: ", best_LAMBDA)
    W_learned = ridge_regression.fit(
        X_train=X_train, Y_train=np.array(Y_train), LAMBDA=best_LAMBDA
    )
    Y_predicted = ridge_regression.predict(W=W_learned, X_new=X_test)

    print(ridge_regression.compute_RSS(Y_new=Y_test, Y_predicted=Y_predicted))
    # print("run now")
