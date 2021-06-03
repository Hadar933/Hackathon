import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_boxplot # add to requirements
import holidays # add to requirements
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
    for
    return df


def main():
    data_dir = r"C:\Users\Owner\Documents\GitHub\IML.HUJI\Hackathon\task1\movies_dataset.csv"
    movies_df = load_data(data_dir)
    look_at_data(movies_df)


if __name__ == '__main__':
    main()