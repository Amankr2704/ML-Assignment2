
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class Boosting:

    def __init__(
        self, X, y, max_depth, n_rounds, random_state=26001
    ):
        self.result = self.train_samme(
            X, y, max_depth, n_rounds, random_state
        )

        # Preserve the original dictionary keys as attributes
        self.trees = self.result["trees"]
        self.alphas = self.result["alphas"]
        self.weighted_errors = self.result["weighted_errors"]
        self.max_sample_weights = self.result["max_sample_weights"]
        self.top10_weight_sums = self.result["top10_weight_sums"]
        self.final_sample_weights = self.result["final_sample_weights"]

    def train_samme(
        self, X, y, max_depth, n_rounds, random_state=26001
    ):
        """
        Train a multiclass SAMME boosting ensemble.
        Returns a dictionary containing trained trees, SAMME weights,
        and per-round diagnostics.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = len(y)
        classes = np.unique(y)
        K = len(classes)

        if K < 2:
            raise ValueError("SAMME requires at least two classes.")
        if n_samples == 0:
            raise ValueError("Training data cannot be empty.")
        if len(X) != n_samples:
            raise ValueError(
                "X and y must contain the same number of samples."
            )
        if n_rounds < 1:
            raise ValueError("n_rounds must be at least 1.")

        # Initialize equal weights for all training samples
        weights = np.full(n_samples, 1.0 / n_samples)
        rng = np.random.default_rng(random_state)

        trees = []
        alphas = []
        weighted_errors = []
        max_sample_weights = []
        top10_weight_sums = []

        random_error_threshold = 1.0 - 1.0 / K

        for t in range(n_rounds):
            tree = DecisionTreeClassifier(
                max_depth=max_depth,
                random_state=int(rng.integers(0, 2**31 - 1))
            )

            # Train the weak learner using current sample weights
            tree.fit(X, y, sample_weight=weights)
            predictions = tree.predict(X)

            incorrect = predictions != y
            weighted_error = float(weights[incorrect].sum())

            # Stop if the learner is no better than random guessing
            if weighted_error >= random_error_threshold:
                break

            # Record diagnostics for the accepted learner
            weighted_errors.append(weighted_error)
            max_sample_weights.append(float(weights.max()))
            top10_weight_sums.append(
                float(np.sort(weights)[-min(10, n_samples):].sum())
            )

            trees.append(tree)

            # Handle a perfect learner with a finite alpha
            if weighted_error <= 1e-12:
                alpha_t = (
                    np.log((1.0 - 1e-12) / 1e-12)
                    + np.log(K - 1)
                )
                alphas.append(float(alpha_t))
                break

            # Calculate the SAMME multiclass learner weight
            alpha_t = (
                np.log((1.0 - weighted_error) / weighted_error)
                + np.log(K - 1)
            )
            alphas.append(float(alpha_t))

            # Increase weights of misclassified samples
            weights[incorrect] *= np.exp(alpha_t)

            # Normalize sample weights
            weights /= weights.sum()

        return {
            "trees": trees,
            "alphas": alphas,
            "weighted_errors": weighted_errors,
            "max_sample_weights": max_sample_weights,
            "top10_weight_sums": top10_weight_sums,
            "final_sample_weights": weights.copy()
        }

    def __getitem__(self, key):
        return self.result[key]


def train_samme(X, y, max_depth, n_rounds, random_state=26001):
    """
    Preserve the original function interface.
    """
    model = Boosting(
        X, y, max_depth, n_rounds, random_state
    )
    return model.result


def predict_samme(model, X):
    """
    Predict classes using SAMME's weighted multiclass voting rule.
    """
    X = np.asarray(X)

    trees = model["trees"]
    alphas = model["alphas"]

    if not trees:
        raise ValueError(
            "SAMME trained no weak learner. "
            "Check the weighted error and training data."
        )

    classes = trees[0].classes_
    class_score = np.zeros(
        (len(X), len(classes)),
        dtype=float
    )

    for tree, alpha in zip(trees, alphas):
        predictions = tree.predict(X)

        for class_index, class_label in enumerate(classes):
            class_score[:, class_index] += (
                alpha * (predictions == class_label)
            )

    return classes[np.argmax(class_score, axis=1)]


def predict_samme_confidence(model, X):
    """
    Return the winning class's fraction of the total weighted vote.
    This is not a calibrated probability.
    """
    X = np.asarray(X)

    trees = model["trees"]
    alphas = model["alphas"]

    if not trees:
        raise ValueError(
            "SAMME trained no weak learner. "
            "Check the weighted error and training data."
        )

    classes = trees[0].classes_
    class_score = np.zeros(
        (len(X), len(classes)),
        dtype=float
    )

    for tree, alpha in zip(trees, alphas):
        predictions = tree.predict(X)

        for class_index, class_label in enumerate(classes):
            class_score[:, class_index] += (
                alpha * (predictions == class_label)
            )

    total_alpha = np.sum(alphas)

    if total_alpha <= 0:
        raise ValueError(
            "The total learner weight must be positive."
        )

    return class_score.max(axis=1) / total_alpha
