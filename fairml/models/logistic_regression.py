from sklearn.linear_model import LogisticRegression


def create_benchmark_logreg(train_X, train_y, max_iter=100):
    train_y = train_y.values.ravel()
    logreg = LogisticRegression(solver="sag", max_iter=max_iter)
    logreg.fit(train_X, train_y)
    print(
        "Mean Accuracy for Training Sets: %.2f %%"
        % (logreg.score(train_X, train_y) * 100)
    )
    return logreg
