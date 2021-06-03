import numpy

import pandas as pd
from sklearn import model_selection
import numpy as np
################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################
#


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """

    #your code goes here...

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

