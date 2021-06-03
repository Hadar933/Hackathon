import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_boxplot
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
    df = df.drop(df[df.status != 'Released'].index)
    return df


def insert_NaNs(df):
    # Replace bad dates with NaNs:
    df = df.mask(df[df.release_date])
    pass


def main():
    data_dir = r"C:\Users\Owner\Documents\GitHub\IML.HUJI\Hackathon\task1\movies_dataset.csv"
    movies_df = load_data(data_dir)
    look_at_data(movies_df)


if __name__ == '__main__':
    main()