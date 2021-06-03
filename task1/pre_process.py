import pandas as pd, codecs, json, heapq
import numpy as np
from plotnine import ggplot, aes, geom_boxplot # add to requirements
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar # add to requirements
from sklearn.feature_extraction.text import CountVectorizer
import datetime
import matplotlib.pyplot as plt
from collections import Counter

# nltk.download('punkt') # Add to requirements
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


def textblob_tokenizer(str_input):
    # For
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words


def add_description_word_features(df):
    """
    make features for words
    :param df:
    :return:
    """
    text_col = df.overview
    txt = text_col.to_list()
    txt = ['missing' if x is np.nan else x for x in txt]
    # nonans_txt = [x for x in txt if str(x) != 'nan']
    # txt = " ".join(nonans_txt)
    # words = txt.split(' ')
    # stem_words = [porter_stemmer.stem(w) for w in words]
    vectorizer = CountVectorizer(stop_words='english', tokenizer=textblob_tokenizer)
    matrix = vectorizer.fit_transform(txt)
    # write in data frame:
    results = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names())
    # Fixme: the join works in a weird way
    df = df.join(results)
    return df


def process_original_langauge(df):
    df.loc[((df.original_language != 'en') &
            (df.original_language != 'fr') &
            (df.original_language != 'hi') &
            (df.original_language != 'ru')), 'original_language'] = 'others'

    df = pd.get_dummies(df, columns=['original_language'])
    return df


def main():
    data_dir = r"C:\Users\Owner\Documents\GitHub\IML.HUJI\Hackathon\task1\movies_dataset.csv"
    movies_df = load_data(data_dir)
    #movies_df = pd.read_pickle('train_old.pkl')
    movies_df = date_col_preprocess(movies_df)
    print(movies_df.shape)
    movies_df = remove_bad_samples(movies_df, 4)
    add_description_word_features(movies_df)
    print(movies_df.shape)


def json_load(json_list, common_vals=None, col_name=None):
    """
    takes a list of string json objects and returns a list of the inside features. e.g. genres or production companies
    :param json_list:
    :param common_vals: for some features there are many options so we store the common or label it other
    :return:
    """
    try:
        list_of_dicts = json.loads(json_list.replace("'", "\""))
        features = [dictio['name'] for dictio in list_of_dicts if not common_vals or dictio['name'] in common_vals]
        if not features and len(list_of_dicts) > 0: features = [f'other-{col_name}']
        return features
    except:
        return np.nan


def jsons_eval():
    train = pd.read_pickle('train.pkl')
    jsons =  [list( map ( json_load , train.genres.values) ),
              list( map ( json_load , train.production_companies.values) ),
              list( map ( json_load , train.production_countries.values) ),
              list( map ( json_load , train.keywords.values) ),
              list( map ( json_load , train.cast.values) ),
              list( map ( json_load , train.crew.values) )]
    jsons_names = ['genres', 'companies', 'countries', 'keywords', 'cast', 'crew']
    f = codecs.open('jsons.txt', 'w', 'utf-8')
    for idx, json_list in enumerate(jsons):
        stats = dict()
        for sample in json_list:
            if sample is not None:
                for entry in sample:
                    name = entry['name']
                    if jsons_names[idx] == 'crew': name += ' dep: ' + entry['known_for_department']
                    if name in stats:
                        stats[name] += 1
                    else:
                        stats[name] = 1
        if len(stats) < 20:
            result = f'-----------------{jsons_names[idx]}---total_vals={len(stats)}--------\n{stats}\n'
        else:
            big20 = dict(Counter(stats).most_common(20))
            result = f'------------------{jsons_names[idx]}-----total-vals={len(stats)}-----best20\n{big20}\n'
        f.write(u'' + result)
    plt.show()
    f.close()


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


def categorical(dataframe):
    """
    creates dummy matrieces from the categorical variables
    :param dataframe:
    :return:
    """
    companies_common = {'Universal Pictures', 'Warner Bros. Pictures', 'Paramount',
                        'Columbia Pictures', '20th Century Fox', 'Metro-Goldwyn-Mayer',
                        'New Line Cinema', 'Walt Disney Pictures'}
    countries_common = {'United States of America', 'United Kingdom', 'France', 'Germany', 'Canada', 'India'}
    genres = list(map(json_load, dataframe.genres.values))
    companies = list(map(lambda p: json_load(p, companies_common, "companies"),
                                                        dataframe.production_companies.values))
    countries = list(map(lambda p: json_load(p, countries_common, 'countries'),
                                                        dataframe.production_countries.values))
    df1 = pd.DataFrame({'genres': genres, 'production_countries': countries,  'production_companies': companies})
    for col in df1:
        df1 = df1.assign(**pd.get_dummies(
            df1[col].apply(lambda x: pd.Series(x)).stack().reset_index(level=1, drop=True)).sum(
            level=0))
        df1 = df1.drop(col, axis=1)
        dataframe = dataframe.drop(col, axis=1)
    return dataframe.join(df1)


def pre_precoss(dataframe):
    """
    performs the pre_process procedure using the sub-functions
    :param dataframe:
    :return:
    """
    df = dataframe.drop(labels=[
    'homepage', 'overview', 'title', 'belongs_to_collection', 'id', 'keywords', 'cast', 'crew',
     'tagline', 'spoken_languages', 'original_title'], axis=1)
    df = remove_not_done_movies(df)
    df = date_col_preprocess(df)
    df = categorical(df)
    df = process_original_langauge(df)
    return df



if __name__ == '__main__':
    train = pd.read_pickle('train.pkl')
    pre_precoss(train)
    #main()