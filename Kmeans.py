def compute_accuracy(predicted_y, expected_y):
    matches =  np.equal(predicted_y, expected_y)
    accuracy = np.sum(matches.astype(float)) / expected_y.size
    return accuracy

def clustering_with_KMeans():
    data, labels = load_data(data_path="../datasets/20news-bydate/20news-full-tfidf.txt")
    from sklearn.cluster import KMeans
    from scipy.sparse import csr_matrix
    X = csr_matrix(data)
    print("==============")
    kmeans = KMeans(
        n_clusters=20,
        init='random',
        n_init=5,
        tol=1e-3,
        random_state=2018
    ).fit(X)
    labels = kmeans.labels_

def classifying_with_linear_SVMs():
    train_X, train_y = load_data(data_path="../datasets/20news-bydate/20news-train-tfidf.txt")
    from sklearn.svm import LinearSVC
    classifier = LinearSVC(
        C=10.0,
        tol=0.001,
        verbose=True
    )
    classifier.fit(train_X, train_y)
    test_X, test_y = load_data(data_path="../datasets/20news-test-tfidf.txt")
    predicted_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y=predicted_y, expected_y=test_y)
    print("Accuracy: ", accuracy)