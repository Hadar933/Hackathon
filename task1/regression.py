################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################
#

import matplotlib.pyplot as plt
import numpy
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor


def RMSE(y, y_hat):
    """
    calculates root mean square error
    :param y: labels
    :param y_hat: prediction
    :return: rmse
    """
    return np.sqrt(np.mean((y_hat - y) ** 2))


def get_Xy1y2_from_pickle(p_file_name):
    """
    extract sample-feature matrix, as well as labels
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
    revenue = df.pop('revenue')
    vote_avg = df.pop('vote_average')
    return df, revenue, vote_avg


def plot_many_models(df, y, regressor_lst):
    for model in regressor_lst:
        print(f"current model = {model}")
        rmse_plot(df, y, model)
    plt.legend(regressor_lst)
    plt.show()


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """

    # your code goes here...

    pass


def rmse_plot(df, y, model):
    """
    plots RMSE error of train and test data for some given model
    :param df: dataframe
    :param y: labels
    :param model: the model in hand
    """
    train, test, y_train, y_test = train_test_split(df, y)  # default is train = 75%, test=25%
    train_size = train.shape[0]
    test_rmse_err = []
    average_y_hat = []  # average
    min_index = 2  # for data < 100, 1% (for example) isn't even one sample, so we give a threshold
    count = 0
    for p in range(min_index, 101):
        count += 1
        # sliced data given some percentage:
        max_index = int((p / 100) * train_size)
        curr_training_set = train.iloc[:max_index, :]
        curr_test_set = test.iloc[:max_index, :]
        curr_y_train = y_train[:max_index]
        curr_y_test = y_test[:max_index]

        # fitting on train, predicting on test
        model.fit(curr_training_set, curr_y_train)
        y_hat = model.predict(curr_test_set)

        # calculating error
        curr_test_rmse_err = RMSE(curr_y_test, y_hat)
        test_rmse_err.append(curr_test_rmse_err)

    plt.title("RMSE values as function of p%")
    plt.xlabel('Percentage (p%)'), plt.ylabel('RMSE')
    percentage = range(min_index, 101)
    plt.plot(percentage, test_rmse_err)


def committee(models):
    """
    generates a learner using committee method
    :param models:
    :return: comittee model (untrained)
    """
    estimators = [(str(model)[:-2], model) for model in models]  # creating estimators input syntax
    return VotingRegressor(estimators=estimators)


def split():
    df = pd.read_csv('movies_dataset.csv')
    train, validate, test = \
        np.split(df.sample(frac=1, random_state=42),
                 [int(.6 * len(df)), int(.8 * len(df))])
    trash_idx = np.zeros((train.shape[0])).astype(numpy.bool)
    trash_idx[[np.random.randint(0, train.shape[0], 100)]] = 1
    trash = train.iloc[trash_idx]
    train = train.iloc[~trash_idx]
    train.to_pickle('train.pkl')
    test.to_pickle('test.pkl')
    validate.to_pickle('valid.pkl')
    trash.to_pickle('trash.pkl')


if __name__ == '__main__':
    X, y1, y2 = get_Xy1y2_from_pickle("trash.pkl")
    X = X[['runtime', 'vote_count', 'budget']]
    gbr = GradientBoostingRegressor()
    rfr = RandomForestRegressor()
    lr = LinearRegression()
    com = committee([gbr, rfr, lr])
    plot_many_models(X, y1, [gbr, rfr, lr, com])

