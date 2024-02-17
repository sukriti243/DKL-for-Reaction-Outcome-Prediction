# adapted from https://github.com/leojklarner/gauche/tree/main/gauche

"""
Abstract class implementing the data loading, data splitting,
type validation and feature extraction functionalities.
"""

from abc import ABCMeta, abstractmethod

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

from drfp import DrfpEncoder

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DataLoader(metaclass=ABCMeta):
    def __init__(self):
        self.task = None

    @property
    @abstractmethod
    def features(self):
        raise NotImplementedError

    @features.setter
    @abstractmethod
    def features(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def labels(self):
        raise NotImplementedError

    @labels.setter
    @abstractmethod
    def labels(self, value):
        raise NotImplementedError

    @abstractmethod
    def validate(self, drop=True):
        """Checks whether the loaded data is a valid instance of the specified
        data type, potentially dropping invalid entries.

        :param drop:  whether to drop invalid entries
        :type drop: bool
        """
        raise NotImplementedError

    @abstractmethod
    def featurize(self, representation):
        """Transforms the features to the specified representation (in-place).

        :param representation: desired feature format
        :type representation: str

        """
        raise NotImplementedError

    def split_and_scale(
        self, test_size=0.2, scale_labels=True, scale_features=False
    ):
        """Splits the data into training and testing sets.

        :param test_size: the relative size of the test set
        :type test_size: float
        :param scale_labels: whether to standardize the labels (after splitting)
        :type scale_labels: bool
        :param scale_features: whether to standardize the features (after splitting)
        :type scale_features: bool
        :returns: (potentially standardized) training and testing sets with associated scalers
        """

        # reshape labels
        self.labels = self.labels.reshape(-1, 1)

        # auxiliary function to perform scaling
        def scale(train, test):
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train)
            test_scaled = scaler.transform(test)
            return train_scaled, test_scaled, scaler

        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=test_size, random_state=1
        )

        # scale features, if requested
        if scale_features:
            features_out = scale(X_train, X_test)
        else:
            features_out = X_train, X_test, None

        # scale labels, if requested
        if scale_labels:
            labels_out = scale(y_train, y_test)
        else:
            labels_out = y_train, y_test, None

        # return concatenated tuples
        return features_out + labels_out

def drfp(reaction_smiles, nBits=2048):
    """
    https://github.com/reymond-group/drfp

    Builds reaction representation as a binary DRFP fingerprints.
    :param reaction_smiles: list of reaction smiles
    :type reaction_smiles: list
    :return: array of shape [len(reaction_smiles), nBits] with drfp featurised reactions

    """
    fps = DrfpEncoder.encode(reaction_smiles, n_folded_length=nBits)
    return np.asarray(fps, dtype=np.float64)


def transform_data(
    X_train, y_train, X_test, y_test, n_components=None, use_pca=False
):
    """
    Apply feature scaling, dimensionality reduction to the data. Return the standardised and low-dimensional train and
    test sets together with the scaler object for the target values.

    :param X_train: input train data
    :param y_train: train labels
    :param X_test: input test data
    :param y_test: test labels
    :param n_components: number of principal components to keep when use_pca = True
    :param use_pca: Whether or not to use PCA
    :return: X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
    """

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    if use_pca:
        pca = PCA(n_components)
        X_train_scaled = pca.fit_transform(X_train)
        print(
            "Fraction of variance retained is: "
            + str(sum(pca.explained_variance_ratio_))
        )
        X_test_scaled = pca.transform(X_test)

    return (
        X_train_scaled,
        y_train_scaled,
        X_test_scaled,
        y_test_scaled,
        y_scaler,
    )

class ReactionLoader(DataLoader):
    def __init__(self):
        super(ReactionLoader, self).__init__()
        self.task = "reaction_yield_prediction"
        self._features = None
        self._labels = None

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    def validate(self, drop=True):
        invalid_idx = []

    def featurize(self, representation, nBits=1024):
        """Transforms reactions into the specified representation.

        :param representation: the desired reaction representation, one of [ohe, rxnfp, drfp, bag_of_smiles]
        :type representation: str
        :param nBits: int giving the bit vector length for drfp representation. Default is 2048
        :type nBits: int
        """

        if representation == "drfp":
            self.features = drfp(self.features.to_list(), nBits=nBits)

        elif representation == "rxnfp":
            self.features = rxnfp(self.features.to_list())

        
        else:
            raise Exception(
                f"The specified representation choice {representation} is not a valid option. "
            )

    def load_benchmark(self, benchmark, path):

        """Loads features and labels from one of the included benchmark datasets
                and feeds them into the DataLoader.

        :param benchmark: the benchmark dataset to be loaded, one of
            ``[DreherDoyle, SuzukiMiyaura, DreherDoyleRXN, SuzukiMiyauraRXN]``
              RXN suffix denotes that csv file contains reaction smiles in a dedicated column.
        :type benchmark: str
        :param path: the path to the dataset in csv format
        :type path: str
        """

        benchmarks = {
            "DreherDoyle": {
                "features": ["ligand", "additive", "base", "aryl halide"],
                "labels": "yield",
            },
            "DreherDoyleRXN": {"features": "rxn", "labels": "yield"},
        }

        if benchmark not in benchmarks.keys():

            raise Exception(
                f"The specified benchmark choice ({benchmark}) is not a valid option. "
                f"Choose one of {list(benchmarks.keys())}."
            )

        else:

            df = pd.read_csv(path)
            # drop nans from the datasets
            nans = df[benchmarks[benchmark]["labels"]].isnull().to_list()
            nan_indices = [nan for nan, x in enumerate(nans) if x]
            self.features = df[benchmarks[benchmark]["features"]].drop(
                nan_indices
            )
            self.labels = (
                df[benchmarks[benchmark]["labels"]]
                .dropna()
                .to_numpy()
                .reshape(-1, 1)
            )


