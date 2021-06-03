################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################
#
import re
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """

    # your code goes here...

    pass


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
    print(f"Current model is {model}")
    X_train, X_test, y_train, y_test = train_test_split(df, y)  # default is train = 75%, test=25%
    train_size = X_train.shape[0]
    test_mse_err, train_mse_err = [], []
    for i in range(1, train_size):
        model.fit(X_train.iloc[:i, :], y_train.iloc[:i])
        y_hat_train = model.predict(X_train.iloc[:i, :])
        y_hat_test = model.predict(X_test)
        test_mse_err.append(mean_squared_error(y_test, y_hat_test))
        train_mse_err.append(mean_squared_error(y_train.iloc[:i], y_hat_train))
    plot_rmse(model, test_mse_err, train_mse_err)


def plot_rmse(model, test_mse_err, train_mse_err):
    """
    plots RMSE error for some given model
    :param model:
    :param test_mse_err:
    :param train_mse_err:
    """
    plt.title(f"{model} RMSE values as function of p%")
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
                X_train, X_test, y_train, y_test = X.iloc[train, :], X.iloc[test, :], y.iloc[train], y.iloc[test]
                model.fit(X_train, y_train)
                y_hat = model.predict(X_test)
                rmse_lst.append(np.sqrt(mean_squared_error(y_test, y_hat)))
            p1p2_mean_err = np.mean(rmse_lst)
            mat[i][j] = p1p2_mean_err
    ind = np.unravel_index(np.argmin(mat, axis=None), mat.shape)
    param1_ind, param2_ind = ind[0], ind[1]
    return depths[param1_ind], min_samples_leaf[param2_ind]


if __name__ == '__main__':
    X, y = get_Xy1y2_from_pickle("trash.pkl", "revenue")
    depth, min_n = regression_tree_k_fold_vc(X, y, [1, 5, 10, 50, 100], [5, 10, 15, 20, 25], 5, RandomForestRegressor)
    rfr = RandomForestRegressor(max_depth=depth,min_samples_leaf=min_n)
    lr = LinearRegression()
    lasso = Lasso(tol=0.001)
    ridge = Ridge(normalize=True)
    com = committee([lasso, ridge, rfr, lr])
    all = [lr, rfr, lasso, ridge, com]
    for m in all:
        test_model_error(X, y, m)
