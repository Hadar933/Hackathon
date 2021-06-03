################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################
#
import re
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """

    # your code goes here...

    pass


def RMSE(y, y_hat):
    """
    calculates root mean square error
    :param y: labels
    :param y_hat: prediction
    :return: rmse
    """
    return np.sqrt(np.mean((y_hat - y) ** 2))


def get_Xy1y2_from_pickle(p_file_name, which_label):
    """
    extract sample-feature matrix, as well as labels
    :param which_label: either revenue or vote average
    :param p_file_name: pickle file name
    :return: X,revenue,vote_average
    """
    objects = []
    with (open(p_file_name, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    df = objects[0]
    df = df[['runtime', 'vote_count', 'budget', 'revenue', 'vote_average']]
    df = df.dropna()
    y = None
    if which_label == 'revenue':
        y = df.pop('revenue')
    elif which_label == 'vote_average':
        y = df.pop('vote_average')
    return df, y


def test_model_error(df, y, model):
    """
    plots RMSE error of train and test data for some given model
    :param df: dataframe
    :param y: labels
    :param model: the model in hand
    """
    train, test, y_train, y_test = train_test_split(df, y)  # default is train = 75%, test=25%
    train_size = train.shape[0]
    test_rmse_err, train_rmse_err = [], []
    min_index = 1  # for data < 100, 1% (for example) isn't even one sample, so we give a threshold
    for p in range(min_index, 101):
        # sliced data given some percentage:
        test_set, train_set, curr_y_test, curr_y_train = slice_data(p, test, train, train_size, y_test, y_train)
        # fitting on train, predicting on test:
        y_hat_test, y_hat_train = fit_and_predict(test_set, train_set, curr_y_train, model)
        # calculating error:
        calculate_error(curr_y_test, curr_y_train, test_rmse_err, train_rmse_err, y_hat_test, y_hat_train)
    plot_rmse(min_index, model, test_rmse_err, train_rmse_err)


def plot_rmse(min_index, model, test_rmse_err, train_rmse_err):
    """
    plots RMSE error for some given model
    :param min_index:
    :param model:
    :param test_rmse_err:
    :param train_rmse_err:
    :return:
    """
    plt.title(f"{model} RMSE values as function of p%")
    plt.xlabel('Percentage (p%)'), plt.ylabel('RMSE')
    percentage = range(min_index, 101)
    plt.plot(percentage, test_rmse_err)
    plt.plot(percentage, train_rmse_err)
    plt.legend(["Test", "Train"])
    plt.show()


def calculate_error(curr_y_test, curr_y_train, test_rmse_err, train_rmse_err, y_hat_test, y_hat_train):
    curr_test_rmse_err = RMSE(curr_y_test, y_hat_test)
    curr_train_rmse_err = RMSE(curr_y_train, y_hat_train)
    test_rmse_err.append(curr_test_rmse_err)
    train_rmse_err.append(curr_train_rmse_err)


def fit_and_predict(curr_test_set, curr_training_set, curr_y_train, model):
    model.fit(curr_training_set, curr_y_train)
    y_hat_test = model.predict(curr_test_set)
    y_hat_train = model.predict(curr_training_set)
    return y_hat_test, y_hat_train


def slice_data(p, test, train, train_size, y_test, y_train):
    max_index = int((p / 100) * train_size)
    curr_training_set = train.iloc[:max_index, :]
    curr_test_set = test.iloc[:max_index, :]
    curr_y_train = y_train[:max_index]
    curr_y_test = y_test[:max_index]
    return curr_test_set, curr_training_set, curr_y_test, curr_y_train


def committee(models):
    """
    generates a learner using committee method
    :param models:
    :return: comittee model (untrained)
    """
    estimators = [(str(model)[:-2], model) for model in models]  # creating estimators input syntax
    return VotingRegressor(estimators=estimators)


def initialize_specific_model(param, modelClass):
    """
    given some param and regression class model, returns an instance of such model with param initialized
    :param param: ex. height of tree
    :param modelClass: some regression class
    :return: instance of class
    """
    if modelClass == RandomForestRegressor:
        return modelClass(max_depth=param)


def k_fold_CV(X, y, hyper_params_lst, K, modelClass):
    """
    performs k-fold cross validation to select desirable Hyper Parameter
    :return:
    """
    kfold = KFold(K, True, 1)
    best_param, best_error = None, np.inf
    for param in hyper_params_lst:
        rmse_lst = []
        for train, test in kfold.split(X):
            model = initialize_specific_model(param, modelClass)
            X_train, X_test, y_train, y_test = X.iloc[train, :], X.iloc[test, :], y.iloc[train], y.iloc[test]
            model.fit(X_train, y_train)
            y_hat = model.predict(X_test)
            rmse_lst.append(RMSE(y_test, y_hat))
        curr_param_error = np.mean(rmse_lst)
        best_param, best_error = (param, curr_param_error) if curr_param_error < best_error else (
            best_param, best_error)
    return best_param


if __name__ == '__main__':
    X, y = get_Xy1y2_from_pickle("trash.pkl", "revenue")
    best_depth = k_fold_CV(X, y, [1, 5, 10, 50, 100], 20, RandomForestRegressor)
    rfr = RandomForestRegressor(max_depth=best_depth)
    lr = LinearRegression()
    lasso = Lasso(tol=0.001)
    ridge = Ridge(normalize=True)
    com = committee([lasso, ridge, rfr, lr])
    all = [lr]
    for m in all:
        test_model_error(X, y, m)
