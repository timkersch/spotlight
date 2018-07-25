"""
Module with functionality for splitting and shuffling datasets.
"""

import numpy as np

from sklearn.utils import murmurhash3_32

from spotlight.interactions import Interactions


def _index_or_none(array, shuffle_index, mask=None):

    if array is None:
        return None
    else:
        if mask is not None:
            array = array[mask]
        return array[shuffle_index]


def shuffle_interactions(interactions,
                         random_state=None):
    """
    Shuffle interactions.

    Parameters
    ----------

    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    interactions: :class:`spotlight.interactions.Interactions`
        The shuffled interactions.
    """

    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(interactions.user_ids))
    random_state.shuffle(shuffle_indices)

    return Interactions(interactions.user_ids[shuffle_indices],
                        interactions.item_ids[shuffle_indices],
                        ratings=_index_or_none(interactions.ratings,
                                               shuffle_indices),
                        timestamps=_index_or_none(interactions.timestamps,
                                                  shuffle_indices),
                        weights=_index_or_none(interactions.weights,
                                               shuffle_indices),
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)


def random_train_test_split(interactions,
                            test_percentage=0.2,
                            random_state=None):
    """
    Randomly split interactions between training and testing.

    Parameters
    ----------

    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    test_percentage: float, optional
        The fraction of interactions to place in the test set.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    (train, test): (:class:`spotlight.interactions.Interactions`,
                    :class:`spotlight.interactions.Interactions`)
         A tuple of (train data, test data)
    """

    interactions = shuffle_interactions(interactions,
                                        random_state=random_state)

    cutoff = int((1.0 - test_percentage) * len(interactions))

    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)

    train = Interactions(interactions.user_ids[train_idx],
                         interactions.item_ids[train_idx],
                         ratings=_index_or_none(interactions.ratings,
                                                train_idx),
                         timestamps=_index_or_none(interactions.timestamps,
                                                   train_idx),
                         weights=_index_or_none(interactions.weights,
                                                train_idx),
                         num_users=interactions.num_users,
                         num_items=interactions.num_items)
    test = Interactions(interactions.user_ids[test_idx],
                        interactions.item_ids[test_idx],
                        ratings=_index_or_none(interactions.ratings,
                                               test_idx),
                        timestamps=_index_or_none(interactions.timestamps,
                                                  test_idx),
                        weights=_index_or_none(interactions.weights,
                                               test_idx),
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)

    return train, test


def time_based_train_test_split(interactions, test_percentage=0.2, test_drop_unknown=True):
    assert interactions.timestamps is not None

    cutoff = int((1.0 - test_percentage) * len(interactions))

    indices = np.argsort(interactions.timestamps)

    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)

    train = Interactions(interactions.user_ids[indices][train_idx],
                         interactions.item_ids[indices][train_idx],
                         ratings=_index_or_none(interactions.ratings, train_idx, indices),
                         timestamps=_index_or_none(interactions.timestamps, train_idx, indices),
                         weights=_index_or_none(interactions.weights, train_idx, indices),
                         num_users=interactions.num_users,
                         num_items=interactions.num_items)

    unkown_item_ids = np.setdiff1d(interactions.item_ids[indices][test_idx], interactions.item_ids[indices][train_idx])

    if test_drop_unknown:
        mask = np.empty(interactions.item_ids[indices][test_idx].shape[0], dtype=bool)
        i = 0
        for x in np.nditer(interactions.item_ids[indices][test_idx]):
            mask[i] = not np.any(unkown_item_ids == x)
            i += 1
    else:
        mask = np.ones(interactions.item_ids[indices][test_idx].shape[0], dtype=bool)

    test = Interactions(interactions.user_ids[indices][test_idx][mask],
                        interactions.item_ids[indices][test_idx][mask],
                        ratings=_index_or_none(_index_or_none(interactions.ratings, test_idx, indices), mask),
                        timestamps=_index_or_none(_index_or_none(interactions.timestamps, test_idx, indices), mask),
                        weights=_index_or_none(_index_or_none(interactions.weights, test_idx, indices), mask),
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)

    return train, test


def user_based_train_test_split(interactions,
                                test_percentage=0.2,
                                random_state=None):
    """
    Split interactions between a train and a test set based on
    user ids, so that a given user's entire interaction history
    is either in the train, or the test set.

    Parameters
    ----------

    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    test_percentage: float, optional
        The fraction of users to place in the test set.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    (train, test): (:class:`spotlight.interactions.Interactions`,
                    :class:`spotlight.interactions.Interactions`)
         A tuple of (train data, test data)
    """

    if random_state is None:
        random_state = np.random.RandomState()

    minint = np.iinfo(np.uint32).min
    maxint = np.iinfo(np.uint32).max

    seed = random_state.randint(minint, maxint, dtype=np.int64)

    in_test = ((murmurhash3_32(interactions.user_ids,
                               seed=seed,
                               positive=True) % 100 /
                100.0) <
               test_percentage)
    in_train = np.logical_not(in_test)

    train = Interactions(interactions.user_ids[in_train],
                         interactions.item_ids[in_train],
                         ratings=_index_or_none(interactions.ratings,
                                                in_train),
                         timestamps=_index_or_none(interactions.timestamps,
                                                   in_train),
                         weights=_index_or_none(interactions.weights,
                                                in_train),
                         num_users=interactions.num_users,
                         num_items=interactions.num_items)
    test = Interactions(interactions.user_ids[in_test],
                        interactions.item_ids[in_test],
                        ratings=_index_or_none(interactions.ratings,
                                               in_test),
                        timestamps=_index_or_none(interactions.timestamps,
                                                  in_test),
                        weights=_index_or_none(interactions.weights,
                                               in_test),
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)

    return train, test
