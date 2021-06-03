import numpy
import pickle
import pandas as pd
import numpy as np
################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################
#


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


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


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """

    # your code goes here...

    pass


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
    X, revenue, vote_avg = get_Xy1y2_from_pickle("trash.pkl")
