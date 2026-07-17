import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.base import clone


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
def cv_comparison(preprocessing_fn, X, y, cv_split, label, classifiers = None, scoring_list = None):
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
        scoring_list = ['accuracy', 'f1', 'roc_auc']
    
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
        print(name, label, 'Results:')
        for metric in scoring_list:
            key = f"test_{metric}"
            print(f"{metric}: {np.mean(results_dict[name][key])}, 'Variance:' {np.var(results_dict[name][key])}")
    
    return results_dict
