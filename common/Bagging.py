
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class Bagging:

    def __init__(self, X_train, y_train, n_estimators, rng):
        self.result = self.bagging(
            X_train, y_train, n_estimators, rng
        )

        # Preserve the original dictionary keys and expose attributes
        self.trees = self.result["trees"]
        self.Data_indices = self.result["Data_indices"]
        self.distinct_counts = self.result["distinct_counts"]

    def bagging(self, X_train, y_train, n_estimators, rng):
        trees = []
        distinct_counts = []
        n_samples_indices = []

        n_samples = X_train.shape[0]

        for i in range(n_estimators):

            # 1. Draw a bootstrap sample of the original training size
            sample_indices = rng.choice(
                X_train.shape[0],
                size=n_samples,
                replace=True
            )

            n_distinct = len(np.unique(sample_indices))
            distinct_counts.append(n_distinct)
            n_samples_indices.append(sample_indices)

            # 2. Generate a reproducible seed for this tree
            tree_seed = rng.integers(0, 2**31 - 1)

            # 3. Train a Decision Tree on the bootstrap sample
            model = DecisionTreeClassifier(
                max_depth=None,
                min_samples_leaf=2,
                random_state=tree_seed
            )

            model.fit(
                X_train[sample_indices],
                y_train[sample_indices]
            )

            trees.append(model)

        return {
            "trees": trees,
            "Data_indices": n_samples_indices,
            "distinct_counts": distinct_counts
        }

    def __getitem__(self, key):
        return self.result[key]


def get_bagging_votes(trees, X):
    """
    Return predictions from every decision tree.
    Shape: (number of samples, number of trees).
    """
    return np.array([
        tree.predict(X) for tree in trees
    ]).T


def predict_bagging(trees, X):
    """
    Predict classes using majority voting.
    """
    preds = get_bagging_votes(trees, X)

    y_pred = np.array([
        np.argmax(np.bincount(votes))
        for votes in preds
    ])

    return y_pred


def evaluate_bagging(
    trees,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test
):
    """
    Evaluate training, validation, and test accuracy.
    """

    train_acc = np.mean(
        y_train == predict_bagging(trees, X_train)
    )

    val_acc = np.mean(
        y_val == predict_bagging(trees, X_val)
    )

    test_acc = np.mean(
        y_test == predict_bagging(trees, X_test)
    )

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc
    }