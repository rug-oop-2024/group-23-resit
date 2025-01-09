from abc import ABC, abstractmethod
import numpy as np

METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "accuracy",
    "precision",
    "r_squared",
    "f1_score",
]


def get_metric(name: str) -> 'Metric':
    """Factory function to get a metric by name.
    Return a metric instance given its str name.
    """
    metrics_dict = {
        "mean_squared_error": MeanSquaredError(),
        "mean_absolute_error": MeanAbsoluteError(),
        "accuracy": Accuracy(),
        "precision": Precision(),
        "r_squared": RSquared(),
        "f1_score": F1Score(),
    }
    return metrics_dict.get(name, None)


class Metric(ABC):
    """Base class for all metrics.
    """

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the metric given ground truth and predictions.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The calculated metric value.
        """
        pass

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Wrapper method to calculate the metric by calling __call__.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The calculated metric value.
        """
        return self.__call__(y_true, y_pred)

# ==========================
# REGRESSION METRICS
# ==========================


class MeanSquaredError(Metric):
    """Mean Squared Error metric implementation.

    Measures the average of the squares of the errors, which is the
    average squared difference between the predicted values and
    the actual values.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the Mean Squared Error (MSE)

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)


class MeanAbsoluteError(Metric):
    """Mean Absolure Error metric implementation

    Measures the average of the absolute differences between the
    predicted values and the actual values.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the Mean Absolute Error (MAE)

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The mean absolute error

        """
        return np.mean(np.abs(y_true - y_pred))


class RSquared(Metric):
    """R-squared (Coefficient of Determination) metric implementation

    R-squared measures the proportion of the variance in the dependent variable
    that is predictable from the independent variable(s). It provides
    an indication of goodness of fit and therefore a measure of how well
    unseen samples are likely to be predicted by the model, through the
    proportion of explained variance.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculate the R-squared value between the true and predicted values

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The R-squared value, representing the proportion of
            variance explained by the model.

        """
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        residual_variance = np.sum((y_true - y_pred) ** 2)
        if total_variance > 0:
            r_squared = 1 - (residual_variance / total_variance)
            return r_squared
        else:
            return 0.0


# ==========================
# CLASSIFICATION METRICS
# ==========================

class Accuracy(Metric):
    """Accuracy metric implementation.

    Measures the proportion of correct predictions made by the
    model out of all predictions.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the Accuracy.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The accuracy as a fraction of correct predictions.
        """
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)
        if total_predictions > 0:
            return correct_predictions / total_predictions
        else:
            return 0.0


class Precision(Metric):
    """Precision metric implementation

    Precision is the ratio of true positive predictions to the total number
    of positive predictions (true positives + false positives).
    It measures the accuracy of the positive predictions.

    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the Precision between the true and predicted labels

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

         Returns:
            float: The precision, representing the ratio of true positives
                   to the sum of true positives and false positives
            It returns 0 if there are no positive predictions
        """
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        false_positive = np.sum((y_true == 0) & (y_pred == 1))
        if (true_positive + false_positive) > 0:
            precision = true_positive / (true_positive + false_positive)
            return precision
        else:
            return 0.0


class F1Score(Metric):
    """F1 Score metric implementation

    The F1 Score is the harmonic mean of precision and recall, providing a
    balance between the two. It is especially useful in situations where
    the class distribution is imbalanced.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the F1 Score between the true and predicted values

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The F1 score, representing the harmonic mean of precision
                   and recall.
            If precision and recall are both zero, it returns 0.
        """
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        false_positive = np.sum((y_true == 0) & (y_pred == 1))
        false_negative = np.sum((y_true == 1) & (y_pred == 0))

        if (true_positive + false_positive) > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0

        if (true_positive + false_negative) > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0

        if (precision + recall) > 0:
            f1_score = (2 * precision * recall) / (precision + recall)
        else:
            f1_score = 0

        return f1_score
