import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_boxplot # add to requirements
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar # add to requirements
from sklearn.feature_extraction.text import CountVectorizer
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
    4. Add how many days has passed from release

    :param df: initial pandas DataFrame
    :return: df - after processing the release date
    """
    # to datetime:
    date_col = pd.to_datetime(df.release_date, errors='coerce')
    df.release_date = date_col

    # Add weekday:
    week_day = [d.weekday() for d in date_col]
    df['week_day'] = week_day
    # Make dummies:
    df = pd.get_dummies(df, columns=['week_day'])

    # Add holidays (now only for US):
    cal = calendar()
    holidays = cal.holidays(start=min(date_col), end = max(date_col))
    df['around_holiday'] = date_col.isin(holidays)
    # Days before holiday
    delta_days = 7
    while delta_days > 0:
        day_ago_col = date_col - datetime.timedelta(days=delta_days)
        df['around_holiday'] = np.logical_or(df['around_holiday'], day_ago_col.isin(holidays))
        delta_days += -1

    # Add how many days has passed:
    today_date = datetime.datetime.now()
    days_num = today_date - date_col
    df['days_from_release'] = days_num.astype('timedelta64[D]')

    # Drop release_date:
    df = df.drop(columns=['release_date'])

    return df


def remove_bad_samples(df, nan_nums):
    """
    Remove samples with more than a certain nan number.
    :param df: pandas dataframe
    :param nan_nums: number of nans we allow
    :return: df pandas dataframe
    """
    df = df[df.isnull().sum(axis=1) < nan_nums]
    return df


def features_to_drop(df, feat_cols):
    """
    removes features by choice.
    :param df: dataframe
    :param feat_cols: what features to delete
    :return: df
    """
    df = df.drop(df, columns=feat_cols)
    return df

def try_txt(df):
    # NOT DONE YET
    text_col = df.overview
    vectorizer = CountVectorizer(stop_words='english')

    txt = text_col.to_list()
    nonans_txt = [x for x in txt if str(x) != 'nan']
    txt = " ".join(nonans_txt)

    voc = vectorizer.fit_transform([txt])


def main():
    data_dir = r"C:\Users\Owner\Documents\GitHub\IML.HUJI\Hackathon\task1\movies_dataset.csv"
    movies_df = load_data(data_dir)
    #movies_df = pd.read_pickle('train_old.pkl')
    movies_df = date_col_preprocess(movies_df)
    print(movies_df.shape)
    movies_df = remove_bad_samples(movies_df, 4)
    try_txt(movies_df)
    print(movies_df.shape)


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
    main()