"""
Disaster response pipeline project
functions to pre-process data : run ETL pipeline that cleans data and stores in a sqlite database

Sample Script Execution:
- To run ETL pipeline that cleans data and stores in database:
    > python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
"""

# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str):
    """
    takes inputs as two csvs, combines them through 'id' and return the combined pandas data frame
    :param messages_filepath: csv file with messages and an unique id
    :param categories_filepath: csv file with categories and unique id
    :return: pandas data frame with merged messages and categories
    """
    # import data from cv
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    - takes the input of pandas dataframe with column named 'categories' with data like below
         ex: related-1;request-0;offer-0;aid_related-0;
    - make a seperate set of columns by splitting from ; and getting the numbers specified there as relevant values
        ex : related | request | offer | aid_related
            1    |   0     |  0    |  0
    - combine it with the df replacing the categories column with these new set of columns
        and drop duplicates
    :param df: python data frame with a column named categories
    :return: df_cleaned : pandas dataframe with categories column splitted by ; and relevant number
    """
    # create a data frame by splitting data by ';' and creating seperate columns for each
    categories = df.categories.str.split(';', expand=True)
    # get the column names for added categories columns by extracting them from the first row of the dataframe
    firstrow = categories.iloc[0, :]
    category_colnames = firstrow.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    # iterate through each column and keep the last character as 1/0
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
        # convert to binary
        categories.loc[categories[column] > 1, column] = 1
    # drop the column - categories and concatenate the cleaned set of splitted columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df_cleaned = df.drop_duplicates()
    return df_cleaned


def save_data(df, database_filename):
    """
    save the clean data set into an sqlite database replacing if exist
    :param df: pandas data frame to save
    :param database_filename: database name with extension .db
            ex: messages.db
    :return: None
    """
    table_name = 'labelled_disaster_messages'
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    """
    main function to run the ETL pipeline
    1. load the csv files
    2. data cleaning
    3. data loading to sqlite database
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
