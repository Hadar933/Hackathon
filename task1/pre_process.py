import ast

import pandas as pd, json
import numpy as np
from plotnine import ggplot, aes, geom_boxplot # add to requirements
#import holidays # add to requirements
import dateutil.parser as dparser
import datetime
import matplotlib.pyplot as plt


def load_data(dir):
    df = pd.read_csv(dir)
    return df

def look_at_data(df):
    features = list(df.columns)
    (
        ggplot(df) # Dataframe
        + aes(x="") # Variables to use
        + geom_boxplot()
    )

def remove_not_done_movies(df):
    # Todo need to *change* and divide into in production/ release that it's likely that the income is zero.
    # Todo divide to a long time ago vs not so long ago
    df = df.drop(df[df.status != 'Released'].index)
    return df


def date_col_preprocess(df):
    """
    1. Change to datetime format and add Nat where problems
    2. Add weekday (NaN where problems)
    3. Add holiday
    :param df:
    :return:
    """
    # to datetime:
    date_col = pd.to_datetime(df.release_date, errors='coerce')
    df.release_date = date_col
    # Add weekday
    week_day = [d.weekday() for d in date_col]
    df[week_day] = week_day
    # Add holidays (now only for US)
    us_holidays = holidays.UnitedStates
    # Days before holiday
    delta_days = 7


def main():
    data_dir = r"C:\Users\Owner\Documents\GitHub\IML.HUJI\Hackathon\task1\movies_dataset.csv"
    movies_df = load_data(data_dir)
    look_at_data(movies_df)


def jsons_eval():
    train = pd.read_pickle('train_old.pkl')
    """jsons = {'genre': train.genres,   'companies' : train.production_companies,
             'countries' : train.production_countries, 'keywords' : train.keywords}
    for json_list in jsons:
        stats = dict()
        for json1 in jsons[json_list]:
            for entry in json.loads(json1):
                if entry in stats:
                    stats[entry] += 1
                else:
                    stats[entry] = 1
        print(f'-------------{json_list}--------------------\n{stats}')"""
    z =  list( map ( lambda x : json.loads(x.replace("'", "\"")) , train.genres.values) )
    x = 2

def split():
    df = pd.concat([pd.read_csv('movies_dataset.csv'), pd.read_csv('movies_dataset_part2.csv')])
    train, validate, test = \
        np.split(df.sample(frac=1, random_state=42),
                 [int(.6 * len(df)), int(.8 * len(df))])
    trash_idx = np.zeros((train.shape[0])).astype(np.bool)
    trash_idx[[np.random.randint(0, train.shape[0], 500)]] = 1
    trash = train.iloc[trash_idx]
    train = train.iloc[~trash_idx]
    train.to_pickle('train.pkl')
    test.to_pickle('test.pkl')
    validate.to_pickle('valid.pkl')
    trash.to_pickle('trash.pkl')
    x=2

if __name__ == '__main__':
    split()