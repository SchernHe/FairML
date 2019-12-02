from sklearn.linear_model import LogisticRegression


def create_benchmark_logreg(train_X, train_y):
    train_y = train_y.values.ravel()
    logreg = LogisticRegression(solver="sag")
    logreg.fit(train_X, train_y)
    print(f"Training Score: {logreg.score(train_X,train_y)}")
    return logreg
