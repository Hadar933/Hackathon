################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
from pre_process import pre_precoss


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """

    # your code goes here...

    pass


def test_model_error(X_train, y_train, X_test, y_test, model):
    """
    plots RMSE error of train and test data for some given model
    :param df: dataframe
    :param y: labels
    :param model: the model in hand
    """
    print(f"Current model is {model}")
    train_size = X_train.shape[0]
    test_mse_err, train_mse_err = [], []
    for i in range(100, train_size, 10):
        model.fit(X_train[:i, :], y_train[:i])
        y_hat_train = model.predict(X_train[:i, :])
        y_hat_test = model.predict(X_test)
        test_mse_err.append(mean_squared_error(y_test, y_hat_test))
        train_mse_err.append(mean_squared_error(y_train[:i], y_hat_train))
    plot_rmse(model, test_mse_err, train_mse_err)


def plot_rmse(model, test_mse_err, train_mse_err):
    """
    plots RMSE error for some given model
    :param model:
    :param test_mse_err:
    :param train_mse_err:
    """
    plt.title(f"{model} RMSE(#samples)")
    plt.xlabel('# samples'), plt.ylabel('RMSE')
    plt.plot(np.sqrt(test_mse_err))
    plt.plot(np.sqrt(train_mse_err))
    plt.legend(["Test", "Train"])
    plt.show()


def committee(models):
    """
    generates a learner using committee method
    :param models:
    :return: comittee model (untrained)
    """
    estimators = [(str(model)[:-2], model) for model in models]  # creating estimators input syntax
    return VotingRegressor(estimators=estimators)


def initialize_specific_model(param1, param2, modelClass):
    """
    given some param and regression class model, returns an instance of such model with param initialized
    :param param1: ex. height of tree
    :param modelClass: some regression class
    :return: instance of class
    """
    if modelClass == RandomForestRegressor:
        return modelClass(max_depth=param1, min_samples_leaf=param2)


def regression_tree_k_fold_vc(X, y, depths, min_samples_leaf, K, modelClass):
    """
    performs k-fold cross validation to select desirable Hyper Parameter for regression tree
    :return: parameters with smallest error
    """
    kfold = KFold(K, True, 1)
    mat = np.zeros((len(depths), len(min_samples_leaf)))
    for i, param1 in enumerate(depths):
        for j, param2 in enumerate(min_samples_leaf):
            rmse_lst = []
            for train, test in kfold.split(X):
                model = initialize_specific_model(param1, param2, modelClass)
                X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
                model.fit(X_train, y_train)
                y_hat = model.predict(X_test)
                rmse_lst.append(np.sqrt(mean_squared_error(y_test, y_hat)))
            p1p2_mean_err = np.mean(rmse_lst)
            mat[i][j] = p1p2_mean_err
    ind = np.unravel_index(np.argmin(mat, axis=None), mat.shape)
    param1_ind, param2_ind = ind[0], ind[1]
    return depths[param1_ind], min_samples_leaf[param2_ind]


if __name__ == '__main__':
    train = pd.read_pickle('train.pkl')
    X, y = pre_precoss(train, "revenue")
    valid = pd.read_pickle('valid.pkl')
    X_test, y_test = pre_precoss(valid, "")
    depths = [1, 5, 10, 50, 100]
    min_samples = [5, 10, 15, 20, 25]
    K = 5
    # best_depth, best_min_n = regression_tree_k_fold_vc(X, y, depths, min_samples, K, RandomForestRegressor)
    # rfr = RandomForestRegressor(max_depth=best_depth, min_samples_leaf=best_min_n)
    rfr = RandomForestRegressor(n_estimators=10)
    lr = LinearRegression()
    # lasso = Lasso(tol=0.001)
    # ridge = Ridge(normalize=True)
    com = committee([lr, rfr])
    all = [lr, rfr, com]
    for m in all:
        test_model_error(X, y, X_test, y_test, m)
