import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def cv_comparison(preprocessing_fn, X, y, cv_split, label, classifiers = None, scoring_list = None):
    """ 
    Performs Cross Validation for XGBoost, Random Forest, Logistic regression given preprocessing steps

    We expect to intake a function that gives us a new list of preprocessing steps when called
    X, y are the data to apply the cross validation on
    cv is the cross validation splitter(e.g. stratified k fold, repeat stratified k fold)
    name is what the outputs should be labeled with
    Classifiers should be a dictionary
    Scoring should be a list
    """

    if scoring_list is None:
        scoring_list = ['accuracy', 'f1']
    
    if classifiers is None:
        classifiers = {
            'XGBoost': XGBClassifier(max_depth = 5, learning_rate = 0.1, random_state = 0),
            'Ramdom Forest': RandomForestClassifier (max_depth = 5, random_state = 0),
            'Logistic Regression': LogisticRegression(max_iter = 1000, random_state = 0)
        }

    for name, model in classifiers.items():
        pipeline = ImbPipeline(steps = preprocessing_fn() + [('classifier', model)])
        results = cross_validate(pipeline, X, y, scoring = scoring_list, cv = cv_split)
        print(name, label, 'Results:')
        for metric in scoring_list:
            key = f"test_{metric}"
            print(f"{metric}: {np.mean(results[key])}, 'Variance:' {np.var(results[key])}")


# This can be optimized even more, where instead of rerunning the whole pipeline for every fold and model, we instead just do it for every fold and then use the same result for all the models.
# This will 1/(# of models) the evaluation time
# However, this is not supported with cross validate so we would have to write the folds part ourselves
    