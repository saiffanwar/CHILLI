"""
Functions for explaining classifiers that use tabular data (matrices).
"""
import collections
import copy
from functools import partial
import json
from threading import local
from typing_extensions import override
import warnings

import numpy as np
import scipy as sp
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from scipy.stats.distributions import norm
from sklearn.linear_model import LinearRegression
from lime_utils.discretize import QuartileDiscretizer
from lime_utils.discretize import DecileDiscretizer
from lime_utils.discretize import EntropyDiscretizer
from lime_utils.discretize import BaseDiscretizer
from lime_utils.discretize import StatsDiscretizer
from . import explanation
from . import lime_base
from pprint import pprint
import random
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pck
#from midasDistances import *
from lime_utils.distances import *
import time
import cProfile as profile
import pstats


class TableDomainMapper(explanation.DomainMapper):
    """Maps feature ids to names, generates table views, etc"""

    def __init__(self, feature_names, feature_values, scaled_row,
                 categorical_features, discretized_feature_names=None,
                 feature_indexes=None):
        """Init.

        Args:
            feature_names: list of feature names, in order
            feature_values: list of strings with the values of the original row
            scaled_row: scaled row
            categorical_features: list of categorical features ids (ints)
            feature_indexes: optional feature indexes used in the sparse case
        """
        self.exp_feature_names = feature_names
        self.discretized_feature_names = discretized_feature_names
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.feature_indexes = feature_indexes
        self.scaled_row = scaled_row
        if sp.sparse.issparse(scaled_row):
            self.all_categorical = False
        else:
            self.all_categorical = len(categorical_features) == len(scaled_row)
        self.categorical_features = categorical_features

    def map_exp_ids(self, exp):
        """Maps ids to feature names.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]

        Returns:
            list of tuples (feature_name, weight)
        """
        names = self.exp_feature_names
        if self.discretized_feature_names is not None:
            names = self.discretized_feature_names
        return [(names[x[0]], x[1]) for x in exp]

    def visualize_instance_html(self,
                                exp,
                                label,
                                div_name,
                                exp_object_name,
                                show_table=True,
                                show_all=False):
        """Shows the current example in a table format.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             show_table: if False, don't show table visualization.
             show_all: if True, show zero-weighted features in the table.
        """
        if not show_table:
            return ''
        weights = [0] * len(self.feature_names)
        for x in exp:
            weights[x[0]] = x[1]
        if self.feature_indexes is not None:
            # Sparse case: only display the non-zero values and importances
            fnames = [self.exp_feature_names[i] for i in self.feature_indexes]
            fweights = [weights[i] for i in self.feature_indexes]
            if show_all:
                out_list = list(zip(fnames,
                                    self.feature_values,
                                    fweights))
            else:
                out_dict = dict(map(lambda x: (x[0], (x[1], x[2], x[3])),
                                zip(self.feature_indexes,
                                    fnames,
                                    self.feature_values,
                                    fweights)))
                out_list = [out_dict.get(x[0], (str(x[0]), 0.0, 0.0)) for x in exp]
        else:
            out_list = list(zip(self.exp_feature_names,
                                self.feature_values,
                                weights))
            if not show_all:
                out_list = [out_list[x[0]] for x in exp]
        ret = u'''
            %s.show_raw_tabular(%s, %d, %s);
        ''' % (exp_object_name, json.dumps(out_list, ensure_ascii=False), label, div_name)
        return ret


class LimeTabularExplainer(object):
    """Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self,
                 training_data,
                 test_data,
                 mode="regression",
                 automated_locality=False,
                 test_labels=None,
                 test_predictions=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='none',
                 discretize_continuous=False,
                 discretizer=None,
                 sample_around_instance=True,
                 random_state=None,
                 training_data_stats=None):
        """Init function.

        Args:
            training_data: numpy 2d array
            mode: "classification" or "regression"
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
                If None, defaults to sqrt (number of columns) * 0.75
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True
                and data is not sparse. Options are 'quartile', 'decile',
                'entropy' or a BaseDiscretizer instance.
            sample_around_instance: if True, will sample continuous features
                in perturbed samples from a normal centered at the instance
                being explained. Otherwise, the normal is centered on the mean
                of the feature data.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            training_data_stats: a dict object having the details of training data
                statistics. If None, training data information will be used, only matters
                if discretize_continuous is True. Must have the following keys:
                means", "mins", "maxs", "stds", "feature_values",
                "feature_frequencies"
        """
        self.random_state = check_random_state(random_state)
        self.mode = mode
        self.categorical_names = categorical_names or {}
        self.sample_around_instance = sample_around_instance
        self.training_data_stats = training_data_stats
        self.test_data = test_data
        self.automated_locality = automated_locality
        self.test_labels = test_labels
        self.test_predictions = test_predictions

        # Check and raise proper error in stats are supplied in non-descritized path
        if self.training_data_stats:
            self.validate_training_data_stats(self.training_data_stats)
        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(training_data.shape[1])]

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        self.discretizer = None
        if discretize_continuous and not sp.sparse.issparse(training_data):
            # Set the discretizer if training data stats are provided
            if self.training_data_stats:
                discretizer = StatsDiscretizer(
                    training_data, self.categorical_features,
                    self.feature_names, labels=training_labels,
                    data_stats=self.training_data_stats,
                    random_state=self.random_state)

            if discretizer == 'quartile':
                self.discretizer = QuartileDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels,
                        random_state=self.random_state)
            elif discretizer == 'decile':
                self.discretizer = DecileDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels,
                        random_state=self.random_state)
            elif discretizer == 'entropy':
                self.discretizer = EntropyDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels,
                        random_state=self.random_state)
            elif isinstance(discretizer, BaseDiscretizer):
                self.discretizer = discretizer
            else:
                raise ValueError('''Discretizer must be 'quartile',''' +
                                 ''' 'decile', 'entropy' or a''' +
                                 ''' BaseDiscretizer instance''')
            self.categorical_features = list(range(training_data.shape[1]))

            # Get the discretized_training_data when the stats are not provided
            if(self.training_data_stats is None):
                discretized_training_data = self.discretizer.discretize(
                    training_data)

        '''########## '''
        if kernel_width is None:
            kernel_width = np.sqrt(training_data.shape[1]) * .75
            # kernel_width = 0.75
        print("Kernel Width: ", kernel_width)
        kernel_width = float(kernel_width)
        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
        self.kernel_fn = partial(kernel, kernel_width=kernel_width)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(self.kernel_fn, verbose, random_state=self.random_state)
        self.class_names = class_names

        # Though set has no role to play if training data stats are provided
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)
        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in self.categorical_features:
            if training_data_stats is None:
                if self.discretizer is not None:
                    column = discretized_training_data[:, feature]
                else:
                    print(feature)
                    column = training_data[:, feature]

                feature_count = collections.Counter(column)
                values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
            else:
                values = training_data_stats["feature_values"][feature]
                frequencies = training_data_stats["feature_frequencies"][feature]

            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1

    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]

    @staticmethod
    def validate_training_data_stats(training_data_stats):
        """
            Method to validate the structure of training data stats
        """
        stat_keys = list(training_data_stats.keys())
        valid_stat_keys = ["means", "mins", "maxs", "stds", "feature_values", "feature_frequencies"]
        missing_keys = list(set(valid_stat_keys) - set(stat_keys))
        if len(missing_keys) > 0:
            raise Exception("Missing keys in training_data_stats. Details: %s" % (missing_keys))

    def explain_instance(self,
                         data_row,
                         instance_num,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=1000,
                         distance_metric='euclidean',
                         model_regressor=None,
                         sampling_method='gaussian',
                         newMethod = True):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array or scipy.sparse matrix, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
            sampling_method: Method to sample synthetic data. Defaults to Gaussian
                sampling. Can also use Latin Hypercube Sampling.

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
            # Preventative code: if sparse, convert to csr format if not in csr format already
            data_row = data_row.tocsr()

        '''Generate Perturbed data around neighbourhod. Creates data which is a bianryu form of the categorical features.'''
        if newMethod:
            perturbations = self.smotePerturbations(data_row, instance_num, num_samples=num_samples)
            # data = self.genPerturbations(data_row, num_samples=2000)
            inverse=perturbations



        if sp.sparse.issparse(perturbations):
            # Note in sparse case we don't subtract mean since data would become dense
            scaled_perturbations = perturbations.multiply(self.scaler.scale_)
            # Multiplying with csr matrix can return a coo sparse matrix
            if not sp.sparse.isspmatrix_csr(scaled_perturbations):
                scaled_perturbations = scaled_perturbations.tocsr()
        else:
            '''Doesnt do anything to categorical features'''

            scaled_perturbations = (perturbations - self.scaler.mean_) / self.scaler.scale_


        ''' Create predictions from full model using raw perturbed data'''
        perturbation_predictions = predict_fn(perturbations)
        self.fullModelPredictions = perturbation_predictions

        if newMethod:
            distances = combinedFeatureDistances(calcAllDistances(data_row, perturbations, self.feature_names))

        else:
            distances = sklearn.metrics.pairwise_distances(
                    scaled_perturbations,
                    scaled_perturbations[0].reshape(1, -1),
                    metric=distance_metric
            ).ravel()




        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(perturbation_predictions.shape) == 1:
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
            elif len(perturbation_predictions.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(perturbation_predictions[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(perturbation_predictions.sum(axis=1), 1.0):
                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(perturbation_predictions.shape)))

        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                if len(perturbation_predictions.shape) != 1 and len(perturbation_predictions[0].shape) == 1:
                    perturbation_predictions = np.array([v[0] for v in perturbation_predictions])
                assert isinstance(perturbation_predictions, np.ndarray) and len(perturbation_predictions.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(perturbation_predictions.shape))

            predicted_value = perturbation_predictions[0]
            min_y = min(perturbation_predictions)
            max_y = max(perturbation_predictions)

            # add a dimension to be compatible with downstream machinery
            perturbation_predictions = perturbation_predictions[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        if sp.sparse.issparse(data_row):
            values = self.convert_and_round(data_row.perturbations)
            feature_indexes = data_row.indices
        else:
            values = self.convert_and_round(data_row)
            feature_indexes = None

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(perturbations.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_perturbations[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names,
                                          feature_indexes=feature_indexes)
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)
        if self.mode == "classification":
            ret_exp.predict_proba = perturbation_predictions[0]
            if top_labels:
                labels = np.argsort(perturbation_predictions[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]

        if newMethod:
            scaled_perturbations = np.array(perturbations)

            # weights = combinedFeatureWeighting(distanceToWeights(distances))
            # weights = distanceToWeightsList(combinedFeatureDistances(distances))

            # weights = [self.kernel_fn(d) for d in weights]
        # else:
        if newMethod == True:
            weights = [self.kernel_fn(d) for d in distances]
        else:
            weights = [self.kernel_fn(d) for d in distances]
        self.weights = weights


        for label in labels:
            (ret_exp.local_model,
             ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label],
             ret_exp.allexpPreds,
             ret_exp.weights) = self.base.explain_instance_with_data(
                    scaled_perturbations,
                    perturbation_predictions,
                    weights,
                    label,
                    num_features,
                    newMethod=newMethod,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp, ret_exp.local_model, perturbations, perturbation_predictions, ret_exp.allexpPreds

    def smotePerturbations(self, instance, instance_num, num_samples=200):
        # Get the list of features and the training data
        features = list(self.feature_names)
        x_test = self.test_data
        y_test = self.test_labels
        y_pred = self.test_predictions

#        x_test = x_test[:2000]
#        y_pred = y_pred[:2000]

        # Start the list of perturbations with the instance being perturbed as first item
        perturbedData = [instance]
        distances = combinedFeatureDistances(calcAllDistances(instance, x_test, features))

        # Reduce the range of the training data set to only include the same clusters as the instance. But all points should be points which satisfy all of the clusters.
        weights = [self.kernel_fn(d) for d in distances]

        # Define some information needed for later:
        dict_x_test = {} # Seperate training data into lists of feature data
        for f, feature in enumerate(features):
            dict_x_test[feature] = [i[f] for i in x_test]
        discreteFeatures = features
        maxVals = [max(val) for val in [dict_x_test[f] for f in dict_x_test.keys()]] # Needed to normalise euclidean distances.
        possValues = [np.unique(val) for val in [dict_x_test[f] for f in dict_x_test.keys()]] # Needed to caculate cyclic distances.
        unique_vals = [np.unique(dict_x_test[f]) for f in features] # Needed to calculate cyclic distances.

        # Calculate distances and weights to assign probabilities when selecting point for interpolation.
        # Distance is calculated in each feature dimension.
        # Loop for number of perturbations required
        for i in range(num_samples):
            perturbation = np.zeros(len(features))
            # Remaining features starts as list of all features to be reduced later.
            remainingFeatures = copy.deepcopy(features)

            interpolationAmount = random.uniform(0,1)
            selectedPoint = random.choices(x_test, weights=weights, k=1)[0]

            # While all features have not been perturbed
            while len(remainingFeatures) != 0:
                # Select a random feature which is yet to be perturbed
                selectedFeature = random.choice(remainingFeatures)
                selectedFeatureIndex = features.index(selectedFeature)

                # Get the value of that feature in the isntance
                instance_value = instance[selectedFeatureIndex]

                # Select a random training data sample based with probability based on the distance to the instance.
                # selectedValue = random.choices((dict_x_test[selectedFeature]), weights=weights, k=1)[0]
                selectedValue = selectedPoint[selectedFeatureIndex]
                # Select a random value between 0 and 1 and adjust the instance value accordingly.
                interpolationDifference = interpolationAmount*(selectedValue - instance_value)
                interpolatedValue = instance_value+interpolationDifference

                # For cyclic features, if interpolated value is in fact further, interpolate in the other direction.
                if calcSingleDistance(instance_value, selectedValue, selectedFeature, maxVals[selectedFeatureIndex], possValues[selectedFeatureIndex]) < calcSingleDistance(instance_value, interpolatedValue, selectedFeature, maxVals[selectedFeatureIndex], possValues[selectedFeatureIndex]):
                    interpolatedValue = instance_value-interpolationAmount

                # If selected feature is discrete, adjust the interpolation so it takes the closest existing value.
                if selectedFeature in discreteFeatures:
                    interpolatedValue = min(unique_vals[selectedFeatureIndex], key=lambda x:abs(x-interpolatedValue))
                # Mark the working feature as perturbed by removing from the list of features yet to be perturbed.
                remainingFeatures.remove(selectedFeature)
                perturbation[selectedFeatureIndex] = interpolatedValue

            # Once all features have been perturbed, add the perturbation to the list of perturbations.
            perturbedData.append(perturbation)

        return perturbedData

class RecurrentTabularExplainer(LimeTabularExplainer):
    """
    An explainer for keras-style recurrent neural networks, where the
    input shape is (n_samples, n_timesteps, n_features). This class
    just extends the LimeTabularExplainer class and reshapes the training
    data and feature names such that they become something like

    (val1_t1, val1_t2, val1_t3, ..., val2_t1, ..., valn_tn)

    Each of the methods that take data reshape it appropriately,
    so you can pass in the training/testing data exactly as you
    would to the recurrent neural network.

    """

    def __init__(self, training_data, mode="classification",
                 training_labels=None, feature_names=None,
                 categorical_features=None, categorical_names=None,
                 kernel_width=None, kernel=None, verbose=False, class_names=None,
                 feature_selection='auto', discretize_continuous=True,
                 discretizer='quartile', random_state=None):
        """
        Args:
            training_data: numpy 3d array with shape
                (n_samples, n_timesteps, n_features)
            mode: "classification" or "regression"
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True. Options
                are 'quartile', 'decile', 'entropy' or a BaseDiscretizer
                instance.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """

        # Reshape X
        n_samples, n_timesteps, n_features = training_data.shape
        training_data = np.transpose(training_data, axes=(0, 2, 1)).reshape(
                n_samples, n_timesteps * n_features)
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        if feature_names is None:
            feature_names = ['feature%d' % i for i in range(n_features)]

        # Update the feature names
        feature_names = ['{}_t-{}'.format(n, n_timesteps - (i + 1))
                         for n in feature_names for i in range(n_timesteps)]

        # Send off the the super class to do its magic.
        super(RecurrentTabularExplainer, self).__init__(
                training_data,
                mode=mode,
                training_labels=training_labels,
                feature_names=feature_names,
                categorical_features=categorical_features,
                categorical_names=categorical_names,
                kernel_width=kernel_width,
                kernel=kernel,
                verbose=verbose,
                class_names=class_names,
                feature_selection=feature_selection,
                discretize_continuous=discretize_continuous,
                discretizer=discretizer,
                random_state=random_state)

    def _make_predict_proba(self, func):
        """
        The predict_proba method will expect 3d arrays, but we are reshaping
        them to 2D so that LIME works correctly. This wraps the function
        you give in explain_instance to first reshape the data to have
        the shape the the keras-style network expects.
        """

        def predict_proba(X):
            n_samples = X.shape[0]
            new_shape = (n_samples, self.n_features, self.n_timesteps)
            X = np.transpose(X.reshape(new_shape), axes=(0, 2, 1))
            return func(X)

        return predict_proba

    def explain_instance(self, data_row, classifier_fn, labels=(1,),
                         top_labels=None, num_features=10, num_samples=1000,
                         distance_metric='euclidean', model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 2d numpy array, corresponding to a row
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities. For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        # Flatten input so that the normal explainer can handle it
        data_row = data_row.T.reshape(self.n_timesteps * self.n_features)

        # Wrap the classifier to reshape input
        classifier_fn = self._make_predict_proba(classifier_fn)
        return super(RecurrentTabularExplainer, self).explain_instance(
            data_row, classifier_fn,
            labels=labels,
            top_labels=top_labels,
            num_features=num_features,
            num_samples=num_samples,
            distance_metric=distance_metric,
            model_regressor=model_regressor)
