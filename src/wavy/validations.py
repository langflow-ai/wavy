import numpy as np
from math import ceil, floor

def _validate_training_split(n_samples, test_size, val_size):
    """
    Validation helper to check if the test/test sizes are meaningful wrt to the
    size of the data (n_samples)
    """

    if test_size is None or val_size is None:
        raise ValueError(
            "test_size and val_size cannot be None"
        )

    test_size_type = np.asarray(test_size).dtype.kind
    val_size_type = np.asarray(val_size).dtype.kind

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

    if val_size is not None and val_size_type not in ("i", "f"):
        raise ValueError(f"Invalid value for val_size: {val_size}")
    if test_size is not None and test_size_type not in ("i", "f"):
        raise ValueError(f"Invalid value for test_size: {test_size}")

    if val_size_type == "f" and test_size_type == "f" and val_size + test_size > 1:
        raise ValueError(f"The sum of test_size and val_size = {val_size + test_size}, should be in the (0, 1) range. Reduce test_size and/or val_size.")


    if test_size_type == "f":
        n_test = ceil(test_size * n_samples)
    elif test_size_type == "i":
        n_test = float(test_size)

    if val_size_type == "f":
        n_val = floor(val_size * n_samples)
    elif val_size_type == "i":
        n_val = float(val_size)

    n_train = n_samples - n_test - n_val

    if n_val + n_test > n_samples:
        raise ValueError(
            "The sum of train_size, val_size and test_size = %d, "
            "should be smaller than the number of "
            "samples %d. Reduce test_size and/or "
            "val_size." % (n_val + n_test, n_samples)
        )

    n_val, n_test, n_train = int(n_val), int(n_test), int(n_train)

    return n_val, n_test, n_train