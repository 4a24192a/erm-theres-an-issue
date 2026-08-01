import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer, fbeta_score, make_scorer
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')


# COMMENT: This is the old version, there were some inefficiencies present here
# This can be optimized even more, where instead of rerunning the whole pipeline for every fold and model, we instead just do it for every fold and then use the same result for all the models.
# This will 1/(# of models) the evaluation time
# However, this is not supported with cross validate so we would have to write the folds part ourselves

# def cv_comparison(preprocessing_fn, X, y, cv_split, label, classifiers = None, scoring_list = None):
#     """ 
#     Performs Cross Validation for XGBoost, Random Forest, Logistic regression given preprocessing steps

#     We expect to intake a function that gives us a new list of preprocessing steps when called
#     X, y are the data to apply the cross validation on
#     cv is the cross validation splitter(e.g. stratified k fold, repeat stratified k fold)
#     name is what the outputs should be labeled with
#     Classifiers should be a dictionary
#     Scoring should be a list
#     """

#     if scoring_list is None:
#         scoring_list = ['accuracy', 'f1']
    
#     if classifiers is None:
#         classifiers = {
#             'XGBoost': XGBClassifier(max_depth = 5, learning_rate = 0.1, random_state = 0),
#             'Ramdom Forest': RandomForestClassifier (max_depth = 5, random_state = 0),
#             'Logistic Regression': LogisticRegression(max_iter = 1000, random_state = 0)
#         }

#     for name, model in classifiers.items():
#         pipeline = ImbPipeline(steps = preprocessing_fn() + [('classifier', model)])
#         results = cross_validate(pipeline, X, y, scoring = scoring_list, cv = cv_split)
#         print(name, label, 'Results:')
#         for metric in scoring_list:
#             key = f"test_{metric}"
#             print(f"{metric}: {np.mean(results[key])}, 'Variance:' {np.var(results[key])}")


# PLANNING NEW VERSION


# We want to ultimately output results, which should be a dictionary, each key is associated with a np array
# the keys are the score methods, the arrays are the results of the score
# The columns of results_model should be the score of the model
# the entries are the scores of the model for a given fold
# We take in all of this information, then we create a preprocessing pipeline
# We get the folds by using cv_split.split(X,y), this should be an array where the first index gives the fold number, 
# and the rest is indices to training set and testing set for the fold
# let's then get X_train, y_train, and X_test and y_test
# we iterate over the folds, for each fold we apply the preprocessing pipeline on X_train and X_test separately
# iterate over models, fit on X_train
# apply multiple_score on X_test, y_test which returns a dictionary with the same keys and whose values are the estimator scores



# estimator, array, array, list -> dict
def multiple_score(estimator,X,y, scoring_list):
    """
    Scores an prefitted estimator based on scoring_llist
    
    Input:
    estimator is a prefitted estimator
    X and y are the data, y is the variable to be predicted
    scoring_list is a list of string names for the scores e.g. ['f1', 'accuracy']

    Returns:
    Dictionary of the scores on a given X and y of a prefitted model with the keys being the scoring method in the form test_score where score is in scoring_list
    """

    results = {}

    for score in scoring_list:
        key = f"test_{score}"
        if score == 'fbeta':
            scorer = make_scorer(fbeta_score, beta = 2) # beta set to 2 here for this problem
        else:
            scorer = get_scorer(score)
        results[key] = scorer(estimator, X, y)
    
    return results

# list of pairs -> list of pairs
def remove_sample_steps(steps_list):
    """
    Removes the last few sampling steps

    Input:
    List of steps, this a list of pairs with first value the name and the second the actual instance
    """
    counter = -1 # index of where we remove up to
    for i, _ in enumerate(steps_list): # i starts from 0 but we want it to actually start from 1
        if hasattr(steps_list[-i-1][-1], 'fit_resample'):
            counter = -i-1
        else:
            return steps_list[:counter]


# fn, array, array, class, string, dict, list -> dict
def cv_comparison(preprocessing_fn, X, y, cv_split, label, classifiers = None, scoring_list = None, output = False):
    """ 
    Performs Cross Validation for XGBoost, Random Forest, Logistic regression given preprocessing steps

    Input: 
    We expect to intake a function that gives us a new list of preprocessing steps when called
    X, y are the data to apply the cross validation on
    cv is the cross validation splitter(e.g. stratified k fold, repeat stratified k fold)
    name is what the outputs should be labeled with
    Classifiers should be a dictionary
    Scoring should be a list

    Returns:
    A dictionary with the estimator names as keys, linked to dictionaries each of which have scoring methods as keys and links to an array of the scores for the models over the folds
    """

    if hasattr(X, 'to_numpy'):
        X = X.to_numpy()

    if hasattr(y, 'to_numpy'):
        y = y.to_numpy()
    if scoring_list is None:
        scoring_list = ['fbeta','average_precision','recall','precision', 'f1']
    
    if classifiers is None:
        classifiers = {
            'XGBoost': XGBClassifier(max_depth = 5, learning_rate = 0.1, random_state = 0),
            'Ramdom Forest': RandomForestClassifier (max_depth = 5, random_state = 0),
            'Logistic Regression': LogisticRegression(max_iter = 1000, random_state = 0)
        }

    total_folds = cv_split.get_n_splits()
    results_dict = {}

    for (name, model) in classifiers.items(): # initiualize the results dictionaries
        results_dict[name] = {} # for each model we assign a results dictionary
        results = results_dict[name]
        for metric in scoring_list:
            key = f"test_{metric}"
            results[key] = np.empty(total_folds) # appending arrays is incredibly inefficient so we first create the dictionary with the keys and empty arrays
            

    for i, (train_index, test_index) in enumerate(cv_split.split(X,y)):
        pipeline = ImbPipeline(steps = preprocessing_fn())
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        if hasattr(pipeline.steps[-1][-1], 'fit_resample'): # We check if there exists a sampler or not. If it is we will need to call resample, recall that preprocessing gives a list of pairs
            pipeline_test = ImbPipeline(steps = remove_sample_steps(pipeline.steps))
            X_train_processed, y_train = pipeline.fit_resample(X_train, y_train)
            X_test_processed = pipeline_test.transform(X_test)
        else:
            X_train_processed = pipeline.fit_transform(X_train, y_train)
            X_test_processed = pipeline.transform(X_test) # transform skips steps like SMOTE, which is intended behaviour for test

        for name, model in classifiers.items():
            results = results_dict[name] # results for this model is found by going through results_dict and finding the dictionary associated with the model
            # results is a dictionary where the keys are the scoring methods and the 

            cloned_model = clone(model) # This is just for good practice, since technically we have been using the same estimators over and over again
            cloned_model.fit(X_train_processed, y_train)
            score_dict = multiple_score(cloned_model, X_test_processed, y_test, scoring_list)
            # this returns a dictionary with the right keys and the values of the scores
            
            for metric in scoring_list:
                key = f"test_{metric}"
                results[key][i] = score_dict[key] # we set the ith entry of a given key to the value under the key in score_dict

    # this loop is separate because we need all of the values before we can compute the mean and variance and print them
    for name, model in classifiers.items():
        print('---',name, label, 'Results:')
        for metric in scoring_list:
            key = f"test_{metric}"
            print(f"{metric}: {np.mean(results_dict[name][key])}, 'Variance:' {np.var(results_dict[name][key])}")
    if output:
        return results_dict


# fn, array, array, class, string, dict, list -> dict
def nested_cv_comparison(preprocessing_fn, X, y, cv_split, label, classifiers=None, scoring_list=None, output=False):
    """
    Nested Cross-Validation that provides unbiased performance estimates by
    treating the entire pipeline (preprocessing + feature selection + model)
    as a single unit.

    The Problem with cv_comparison:
    In cv_comparison, the preprocessing pipeline (including feature selection
    via SelectFromModel) is fit on the training fold, then a separate model is
    trained on the transformed training fold and evaluated on the transformed
    test fold. While this is technically correct per fold, the CV estimate can
    be optimistic because:
      - All folds are drawn from the same X_train pool
      - Feature selection adapts to distributional characteristics shared
        across the pool, which inflates scores for data from that same pool
      - A truly held-out test set does not share these characteristics, so
        performance drops when evaluated on it

    How nested_cv_comparison fixes this:
    Instead of separating preprocessing from the model, we build a complete
    ImbPipeline that includes the model as the final step. We then use
    sklearn's cross_validate on this full pipeline. This means:
      - The entire pipeline (imputation, scaling, feature selection, sampling,
        AND the classifier) is treated as one estimator
      - cross_validate handles the fold splitting, fitting, and scoring
      - Feature selection never sees the test fold, not even indirectly
      - The resulting scores are a more realistic estimate of generalization

    Input:
    preprocessing_fn: A function that returns a list of (name, transformer)
        tuples. This should include ALL steps including the model/classifier
        as the last step. If the last step is a sampler (has fit_resample),
        then the model should be the second-to-last non-sampler step or
        included after sampling steps (the ImbPipeline handles this).
    X, y: the data to apply cross validation on
    cv_split: cross validation splitter (e.g. RepeatedStratifiedKFold)
    label: string label for printing results
    classifiers: dictionary of {name: (model, preprocessing_fn)} pairs.
        Each preprocessing_fn should return steps that include the model.
        If None, uses default classifiers with standard preprocessing.
    scoring_list: list of scoring metric names
    output: if True, returns the results dictionary

    Returns:
    A dictionary with estimator names as keys, linked to dictionaries each
    of which have scoring methods as keys linked to arrays of fold scores.
    """
    if hasattr(X, 'to_numpy'):
        X = X.to_numpy()
    if hasattr(y, 'to_numpy'):
        y = y.to_numpy()

    if scoring_list is None:
        scoring_list = ['fbeta', 'average_precision', 'recall', 'precision', 'f1']

    # Build the scoring dictionary that cross_validate expects
    scoring_dict = {}
    for metric in scoring_list:
        if metric == 'fbeta':
            scoring_dict[metric] = make_scorer(fbeta_score, beta=2)
        else:
            scoring_dict[metric] = metric

    # If classifiers is None, use defaults with full pipeline (preprocessing + model)
    if classifiers is None:
        classifiers = {
            'XGBoost': XGBClassifier(max_depth=5, learning_rate=0.1, random_state=0),
            'Random Forest': RandomForestClassifier(max_depth=5, random_state=0),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=0)
        }

    results_dict = {}

    for name, model in classifiers.items():
        # Build the full pipeline: preprocessing steps + classifier
        # The key difference from cv_comparison: the model is INSIDE the pipeline
        # so cross_validate treats the whole thing (feature selection + model) as one unit
        full_pipeline = ImbPipeline(
            steps=preprocessing_fn() + [('classifier', clone(model))]
        )

        # cross_validate handles:
        # 1. Splitting into train/test folds
        # 2. Fitting the ENTIRE pipeline (including feature selection) on train only
        # 3. Scoring on the test fold (which was never seen during any fitting step)
        cv_results = cross_validate(
            full_pipeline, X, y,
            scoring=scoring_dict,
            cv=cv_split,
            n_jobs=-1,
            error_score='raise'
        )

        # Reformat results to match cv_comparison output format
        results_dict[name] = {}
        for metric in scoring_list:
            key = f"test_{metric}"
            results_dict[name][key] = cv_results[f"test_{metric}"]

    # Print results in the same format as cv_comparison
    for name in classifiers.keys():
        print('---', name, label, 'Results (Nested CV):')
        for metric in scoring_list:
            key = f"test_{metric}"
            scores = results_dict[name][key]
            print(f"{metric}: {np.mean(scores):.4f}, Variance: {np.var(scores):.4f}")

    if output:
        return results_dict


# We code Stability Selection

# Stability Selection is a feature selection method
# We will try to match the syntax with SKLearn(we will choose to not include some methods)
# This means that StabilitySelection should be a class
# Stability selection will be class that holds BaseEstimator and TransformerMixin(this is just convention)

# Stability selection takes in an estimator(default should be LASSO), the number of resamples to take(n_bootstrap)
# the size of each of the samples(sample_fraction)
# ========================================
# side note: we use bootstrapping here which is with replacement to mimic a sample of a large population. We want to see how sampling and chance would affect
# the parameters chosen in this case, so if we used without replacement, we wouold get very similar samples that do not appear to be from a theoretically
# large or nearly infinite population
# ========================================
# as well as the threshold to include a variable

# Stability selection works by taking many bootstrapped random samples, then looking at which ones get used by LASSO(which will automatically eliminate irrelevant features)
# In the end, we calculate the probability that a certain variable was used, if it surpasses a certain amount(the threshold), then we include it

# The reason why stability selection works is that we are essentially filtering out the noise. We take bootstrapped samples to simulate
# sample a large population, and we look at how often a certain variable actually matters in this sample
# if it is often enough, then we include it. Otherwise it was just random and thus noise


# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.utils.validation import check_is_fitted, check_array
# from sklearn.linear_model import LogisticRegression
# from sklearn.utils import resample
# import numpy as np

# class StabilitySelection(BaseEstimator, TransformerMixin):
#     """
#     A class for Stability Selection

#     Args
#     Estimator: where feature importance is derived from
#     n_bootstrap: number of iterations
#     sample_fraction: size of the sample relative to X
#     threshold: minimum importance probability to get included
#     random_state: random state seed
#     """

#     def __init__(self, estimator=None, n_bootstrap=100, sample_fraction=0.5, threshold=0.6, random_state=0): # we first initialize all of the variables
#         self.estimator = estimator
#         self.n_bootstrap = n_bootstrap
#         self.sample_fraction = sample_fraction
#         self.threshold = threshold
#         self.random_state = random_state

#     def fit(self, X, y): # this should return itself with certain attributes defined, features_selected, selection_frequency
#         """
#         A class method to fit the stability selection transformer

#         Args
#         X: the data of input variables
#         y: the data of target variable(s)
#         """
#         X = check_array(X)
#         n_samples, n_features = X.shape
#         sample_fraction = self.sample_fraction
#         n_iterations = self.n_bootstrap
#         rng = np.random.RandomState(seed = self.random_state) # we need to have a seeded rng for the subsamples
#         estimator = self.estimator or LogisticRegression(penalty = 'l1', solver = 'liblinear', random_state = self.random_state)
#         appearance_count = np.zeros(n_features) # We also need to count how many times each feature appeared

        
#         for i in range(0,n_iterations):
#             X_resampled, y_resampled = resample(X, y, n_samples = int(n_samples * sample_fraction), random_state=rng.randint(0, 1000000), stratify = y)
#             model = clone(estimator)
#             model.fit(X_resampled,y_resampled)
#             coef = model.coef_.ravel()
#             coef = (coef != 0).astype(int) # applies != 0 on coef, then converts to int
#             appearance_count += coef # vectorized operations are faster than looping and iterating through

#         selection_frequency = appearance_count/n_iterations # gives probability of appearance
#         self.selection_frequency_ = selection_frequency
#         self.features_selected_ = np.where(selection_frequency >= self.threshold) # returns the indices of the selected features
#         self.n_features_in_ = n_features # number of features in 
#         return self

#     def transform(self, X): # this should return X with only the selected features
#         """
#         Applying Stability Selection once the transformer is fitted
        
#         Args
#         X: data of input variables
#         """
#         check_is_fitted(self, 'features_selected_') # checks if features_selected_ appears, otherwise raises an error
#         check_array(X)
#         return X[:, self.features_selected_]

#     def get_feature_names_out(self, input_features = None): # this should return the names of the features. If not provided, then return the indices
#         """
#         Returning the selected feature names of a fitted Stability Selection Transformer

#         Args
#         input_features: List of input feature names
#         """
#         check_is_fitted(self, 'features_selected_')
#         if input_features is None:
#             input_features = [f"x{i}" for i in range(self.n_features_in_)]
#         return np.array(input_features)[self.features_selected_]



            

# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.utils.validation import check_is_fitted, check_array
# from sklearn.linear_model import LogisticRegression
# from sklearn.utils import resample
# import numpy as np

# class StabilitySelection(BaseEstimator, TransformerMixin):
#     def __init__(self, estimator=None, n_bootstrap=100, sample_fraction=0.75,
#                  threshold=0.6, random_state=None):
#         self.estimator = estimator
#         self.n_bootstrap = n_bootstrap
#         self.sample_fraction = sample_fraction
#         self.threshold = threshold
#         self.random_state = random_state

#     def fit(self, X, y):
#         X = check_array(X)
#         n_samples, n_features = X.shape
#         base_estimator = self.estimator or LogisticRegression(
#             penalty='l1', solver='liblinear', random_state=self.random_state
#         )
#         rng = np.random.RandomState(self.random_state)
#         selection_counts = np.zeros(n_features)

#         for i in range(self.n_bootstrap):
#             X_resampled, y_resampled = resample(
#                 X, y, n_samples=int(n_samples * self.sample_fraction),
#                 random_state=rng.randint(0, 1_000_000), stratify=y
#             )
#             model = clone(base_estimator)
#             model.fit(X_resampled, y_resampled)
#             coefs = model.coef_.ravel()
#             selection_counts += (np.abs(coefs) > 1e-8).astype(int)

#         self.selection_frequency_ = selection_counts / self.n_bootstrap
#         self.selected_features_ = np.where(self.selection_frequency_ >= self.threshold)[0]
#         self.n_features_in_ = n_features
#         return self

#     def transform(self, X):
#         check_is_fitted(self, 'selected_features_')
#         X = check_array(X)
#         return X[:, self.selected_features_]

#     def get_feature_names_out(self, input_features=None):
#         check_is_fitted(self, 'selected_features_')
#         if input_features is None:
#             input_features = [f"x{i}" for i in range(self.n_features_in_)]
#         return np.array(input_features)[self.selected_features_]


"""
Stability Selection (Meinshausen & Bühlmann, 2010)

Fits L1-regularized logistic regression across a range of regularization
strengths ("stability path"), on many bootstrap resamples of the data.
A feature's stability score is the MAXIMUM selection frequency it achieves
across that entire regularization path -- not its frequency at one fixed
strength. This is what distinguishes proper stability selection from a
simpler "bootstrap-aggregated Lasso at one C" heuristic, and is what gives
the method its theoretical error-control properties.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.utils.validation import check_is_fitted, check_array


class StabilitySelection(BaseEstimator, TransformerMixin):
    """
    Select features that are consistently chosen by L1-regularized logistic
    regression across many bootstrap resamples and a range of regularization
    strengths.

    Args
    estimator : estimator object, optional
        Base estimator to fit on each bootstrap. Must expose `coef_` after
        fitting (i.e. a linear model). Defaults to L1-penalized
        LogisticRegression with the 'liblinear' solver.
        note: regularization strength of the estimator is ignored, we sweep
        C values manually

    n_bootstrap : int, default=100
        Total number of bootstrap fits. Split evenly across each value in
        `Cs`, so the per-C bootstrap count is n_bootstrap // len(Cs).

    sample_fraction : float, default=0.75
        Fraction of the training data drawn with stratification in each
        bootstrap resample.

    Cs : array-like, default=np.logspace(-2, 1, 10)
        The regularization path to sweep. Smaller C means stronger penalty
        which means more zeroes. The default spans a wide range
        which can be narrowed once we've inspected where the data's
        selection frequencies actually separate

    threshold : float, default=0.6
        A feature is selected if its maximum selection frequency across
        the Cs path meets or exceeds this value. The original paper's
        theoretical guarantees hold for thresholds roughly in the 0.6-0.9
        range. This assumes the selection frequencies are
        well-separated i.e. bimodal to begin with. If they aren't
        (check via `plot_stability_path`), no threshold value in this
        range is "correct," since you're not really choosing
        between two separated groups.

    random_state : int, optional
        Seed for reproducibility. Each bootstrap draws its own derived
        seed from this master seed, so resamples are different
        from each other while the whole run stays reproducible.

    Attributes (set after fit)
    selection_path_ : ndarray of shape (n_features, len(Cs))
        Selection frequency of each feature at each C value individually
        useful for diagnosing whether your Cs range is well-chosen.

    selection_frequency_ : ndarray of shape (n_features,)
        Maximum selection frequency per feature across the whole Cs path.

    selected_features_ : ndarray
        Integer column indices of features meeting `threshold`.

    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(self, estimator=None, n_bootstrap=100, sample_fraction=0.75,
                 Cs=None, threshold=0.6, random_state=None):
        self.estimator = estimator
        self.n_bootstrap = n_bootstrap
        self.sample_fraction = sample_fraction
        self.Cs = Cs
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        base_estimator = self.estimator or LogisticRegression(
            penalty='l1', solver='liblinear', random_state=self.random_state
        )
        Cs = self.Cs if self.Cs is not None else np.logspace(-2, 1, 10)

        rng = np.random.RandomState(self.random_state)
        n_iter_per_C = max(1, self.n_bootstrap // len(Cs))

        # Per-C selection counts, so we can inspect the path afterward
        # and diagnose whether Cs was well-chosen not just the final
        # max-aggregated result.
        selection_path = np.zeros((n_features, len(Cs)))

        for c_idx, C in enumerate(Cs):
            counts = np.zeros(n_features)
            for _ in range(n_iter_per_C):
                X_resampled, y_resampled = resample(
                    X, y,
                    n_samples=int(n_samples * self.sample_fraction),
                    random_state=rng.randint(0, 1_000_000),
                    stratify=y  # preserves class ratio in every bootstrap --
                                # important given how few minority examples
                                # exist, an unstratified draw could easily
                                # produce a bootstrap with even fewer
                )
                model = clone(base_estimator)
                model.set_params(C=C)  # override whatever C the passed
                                        # estimator had, we're sweeping it
                model.fit(X_resampled, y_resampled)

                coefs = model.coef_.ravel()
                # 1e-8 tolerance, not `!= 0`: L1 conceptually zeros
                # coefficients, but floating point arithmetic rarely
                # lands on an exact zero, so a strict != 0 check would
                # miss coefficients that are effectively zero.
                counts += (np.abs(coefs) > 1e-8).astype(int)

            selection_path[:, c_idx] = counts / n_iter_per_C

        # The core stability selection idea: a feature's stability score
        # is its BEST (max) showing anywhere along the regularization path,
        # not its score at one arbitrary fixed C. A feature that's reliably
        # selected at even one sensible regularization strength counts as
        # stable
        self.selection_path_ = selection_path
        self.selection_frequency_ = selection_path.max(axis=1)
        self.selected_features_ = np.where(self.selection_frequency_ >= self.threshold)[0]
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        check_is_fitted(self, 'selected_features_')
        X = check_array(X)
        return X[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'selected_features_')
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.array(input_features)[self.selected_features_]

    def plot_stability_path(self, feature_names=None, top_n=20):
        """
        Diagnostic plot: shows each feature's selection frequency across
        the Cs sweep. Use this before trusting `threshold`. If the top
        features show a clean plateau near 1.0 that clearly separates from
        a cluster near 0.0, your threshold choice barely matters. If
        instead frequencies are smoothly spread with no separation, that's
        a sign either Cs needs adjusting, or the data doesn't
        support confident feature selection at your sample size
        """
        check_is_fitted(self, 'selection_path_')
        import matplotlib.pyplot as plt

        Cs = self.Cs if self.Cs is not None else np.logspace(-2, 1, 10)
        top_idx = np.argsort(self.selection_frequency_)[::-1][:top_n]

        plt.figure(figsize=(8, 5))
        for idx in top_idx:
            label = feature_names[idx] if feature_names is not None else f"x{idx}"
            plt.plot(Cs, self.selection_path_[idx], alpha=0.6, label=label)
        plt.xscale('log')
        plt.xlabel('C (regularization strength)')
        plt.ylabel('Selection frequency')
        plt.title(f'Stability path — top {top_n} features by max frequency')
        plt.axhline(self.threshold, color='red', linestyle='--', label=f'threshold={self.threshold}')
        plt.legend(fontsize=7, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()

        # print the overall distribution shape as a quick numeric check
        freq = self.selection_frequency_
        print(f"Selection frequency distribution:")
        print(f"  < 0.2: {np.sum(freq < 0.2)} features")
        print(f"  0.2-0.6: {np.sum((freq >= 0.2) & (freq < 0.6))} features  <- ambiguous zone")
        print(f"  >= 0.6: {np.sum(freq >= 0.6)} features")


