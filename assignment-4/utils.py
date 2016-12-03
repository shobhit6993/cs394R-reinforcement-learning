import numpy as np


def normalize_probabilities(probabilities):
    """Normalizes the probabilities so that they sum to one.

    If they don't -- due to precision issues -- the difference is added to
    some element until it does.

    Args:
        probabilities (numpy.ndarray): Vector of probabilities.
        s (float, optional): Description
    """
    sum_of_probabilities = np.sum(probabilities)
    while sum_of_probabilities != 1.0:
        diff = 1.0 - sum_of_probabilities
        random_index = np.random.randint(len(probabilities))
        new_value = probabilities[random_index] + diff
        if new_value >= 0 and new_value <= 1:
            probabilities[random_index] += diff
        sum_of_probabilities = np.sum(probabilities)


def state_value(feature_vector, theta):
    """Computes the state-value using the linear function approximation.

    Args:
        feature_vector (numpy.ndarray): Feature vector for a state.
        theta (numpy.ndarray): Parameters characterizing the linear fucntion
            approximator.

    Returns:
        float: Value of the state.
    """
    return np.dot(feature_vector, theta)


def rmse(predictions, targets):
    """Computes Root-Mean-Squared Error between the target and prediction.

    Args:
        predictions (numpy.ndarray): Vector of predictions.
        targets (numpy.ndarray): Vector of targets.

    Returns:
        float: RMSE between predicitons and target.
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())
