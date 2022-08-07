from typing import Tuple

import numpy as np


def _validate_training_split(
    n_samples: int, train_size: int, val_size: int, test_size: int
) -> Tuple[int, int, int]:
    """
    Validation helper to check if the test/test sizes are meaningful wrt to the
    size of the data (n_samples)

    Args:
        n_samples (int): Number of samples in the data
        train_size (int): Size of the training set
        val_size (int): Size of the validation set
        test_size (int): Size of the test set

    Returns:
        Tuple[int, int, int]: Tuple of the training, validation, and test sizes
    """

    if sum(x is None for x in (train_size, val_size, test_size)) > 1:
        raise ValueError(
            "At least one from train_size, val_size and test_size can be None"
        )

    train_size_type = np.asarray(train_size).dtype.kind
    test_size_type = np.asarray(test_size).dtype.kind
    val_size_type = np.asarray(val_size).dtype.kind

    # Check train_size
    if (
        train_size_type == "i"
        and (train_size >= n_samples or train_size <= 0)
        or train_size_type == "f"
        and (train_size <= 0 or train_size >= 1)
    ):
        raise ValueError(
            "train_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(train_size, n_samples)
        )

    # Check val_size
    if (
        val_size_type == "i"
        and (val_size >= n_samples or val_size <= 0)
        or val_size_type == "f"
        and (val_size <= 0 or val_size >= 1)
    ):
        raise ValueError(
            "val_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(val_size, n_samples)
        )

    # Check test_size
    if (
        test_size_type == "i"
        and (test_size >= n_samples or test_size <= 0)
        or test_size_type == "f"
        and (test_size <= 0 or test_size >= 1)
    ):
        raise ValueError(
            "test_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(test_size, n_samples)
        )

    if train_size is not None and train_size_type not in ("i", "f"):
        raise ValueError(f"Invalid value for train_size: {train_size}")
    if test_size is not None and test_size_type not in ("i", "f"):
        raise ValueError(f"Invalid value for test_size: {test_size}")
    if val_size is not None and val_size_type not in ("i", "f"):
        raise ValueError(f"Invalid value for val_size: {val_size}")

    # Check if percentages sum to less than 1
    if (
        train_size_type == "f"
        and val_size_type == "f"
        and test_size_type == "f"
        and train_size + val_size + test_size > 1
    ):
        raise ValueError(
            f"""The sum of train_size, val_size and test_size =
            {train_size + val_size + test_size}, should be in the (0, 1) range.
            Reduce train_size and/or val_size and/or test_size."""
        )

    if train_size_type == "f":
        n_train = round(train_size * n_samples)
    elif train_size_type == "i":
        n_train = float(train_size)

    if val_size_type == "f":
        n_val = round(val_size * n_samples)
    elif val_size_type == "i":
        n_val = float(val_size)

    if test_size_type == "f":
        n_test = round(test_size * n_samples)
    elif test_size_type == "i":
        n_test = float(test_size)

    # Calculate the number of samples in each split if any is None
    if not train_size:
        n_train = max(n_samples - n_val - n_test, 0)
    elif not val_size:
        n_val = max(n_samples - n_train - n_test, 0)
    elif not test_size:
        n_test = max(n_samples - n_train - n_val, 0)

    if abs(n_train + n_val + n_test - n_samples) == 1:
        n_test = max(n_samples - n_train - n_val, 0)

    if n_train + n_val + n_test != n_samples:
        raise ValueError(
            f"""The sum of train_size, val_size and test_size = {n_train + n_val + n_test},
            should be of the same size of samples {n_samples}.
            Change train_size and/or val_size and/or test_size."""
        )

    return int(n_train), int(n_val), int(n_test)


def _validate_sample_panel(
    samples: int, train_size: int, val_size: int, test_size: int
) -> Tuple[int, int, int]:
    """
    Validation helper to check if the samples size is meaningful wrt to the
    size of the data (n_samples)

    Args:
        samples (int): Number of samples in the data
        train_size (int): Size of the training set
        val_size (int): Size of the validation set

    Returns:
        Tuple[int, int, int]: Tuple of the training, validation, and test sizes
    """

    n_samples = train_size + val_size + test_size
    samples_type = np.asarray(samples).dtype.kind

    # Check samples size
    if (
        samples_type == "i"
        and (samples >= n_samples or samples <= 0)
        or samples_type == "f"
        and (samples <= 0 or samples >= 1)
    ):
        raise ValueError(
            "samples={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(samples, n_samples)
        )

    if samples is not None and samples_type not in ("i", "f"):
        raise ValueError(f"Invalid value for samples: {samples}")

    if samples_type == "i":
        samples /= n_samples

    train_samples = round(samples * train_size)
    val_samples = round(samples * val_size)
    test_samples = round(samples * test_size)

    return train_samples, val_samples, test_samples
