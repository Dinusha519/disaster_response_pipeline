"""
Disaster response pipeline project
functions to train the model on preprocessed data

Sample Script Execution:
> python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
when providing the database path please provide the folder which contains all folders as root or else provide the absolute paths
"""
import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import fbeta_score, classification_report
from scipy.stats.mstats import gmean
from sklearn.ensemble import AdaBoostClassifier



def load_data(database_filepath):
    """
    load the data from the data base
    input:
        database_filepath: database path as a string
    output:
        x : feature set
        y : label set
        category_names: names of labels
    """
    table_name = 'labelled_disaster_messages'
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, engine)
    x = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns
    return x, y, category_names


def tokenize(text):
    """
    Tokenize function

    input:
        text -> list of text messages (english)
    Output:
        clean_tokens -> tokenized text, clean for ML modeling
    """
    # Replace the urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Extract the urls from the text and replacing with urlplaceholder
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize and lemmaize text
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()

    # clean tokens by stipping and putting them to lowercase
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_pipeline():
    """
    building a pipeline to process the text data including
    - CountVectorizer
    - TfidfTransformer
    - StartingVerbExtractor
    - classifier

    output:
        pipeline to apply to text which process and apply a classifier
    """
    pipeline = Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer()),
                ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
            ])
    return pipeline


def evaluate_pipeline(pipeline, X_test, Y_test, category_names):
    """
    evaluate the model performance with accuracy and F1 score
    input:
        pipeline -> scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> Test label names
    """
    # predict
    Y_pred = pipeline.predict(X_test)

    # calculate F1 score and accuracy
    multi_f1 = multioutput_fscore(Y_test, Y_pred, beta=1)
    overall_accuracy = (Y_pred == Y_test).mean().mean()

    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy * 100))
    print('F1 score (custom definition) {0:.2f}%'.format(multi_f1 * 100))

    # print the full classification report.
    Y_pred = pd.DataFrame(Y_pred, columns=Y_test.columns)

    for column in Y_test.columns:
        print('Model Performance - Category: {}'.format(column))
        print(classification_report(Y_test[column], Y_pred[column]))


def multioutput_fscore(y_true, y_pred, beta=1):
    """
    F1 score compatible with multi label and muli class problems
    input:
        y_true - List of labels
        y_pred - List of predictions
        beta - Beta value to be used to calculate fscore metric

    Output:
        f1score - geometric mean of fscore
    """

    # If provided y predictions is a dataframe then extract the values from that
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values

    # If provided y actuals is a dataframe then extract the values from that
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values

    f1score_list = []
    for column in range(0, y_true.shape[1]):
        score = fbeta_score(y_true[:, column], y_pred[:, column], beta, average='weighted')
        f1score_list.append(score)

    f1score = np.asarray(f1score_list)
    f1score = f1score[f1score < 1]

    # Get the geometric mean of f1score
    f1score = gmean(f1score)
    return f1score


def save_model_as_pickle(pipeline, pickle_filepath):
    """
    Save the model as a pickle
    input:
        pipeline: Scikit Pipeline object
        pickle_filepath -> destination path to save .pkl file

    """
    pickle.dump(pipeline, open(pickle_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data from the databse : {} ...'.format(database_filepath))

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print('Preparing the data for modelling...')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Building the pipeline ...')
        pipeline = build_pipeline()
        print('Training data with the built pipeline ...')
        pipeline.fit(X_train, Y_train)
        print('Evaluating the model...')
        evaluate_pipeline(pipeline, X_test, Y_test, category_names)
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model_as_pickle(pipeline, model_filepath)
    else:
        print("Please provide the arguments correctly: \nSample Script Execution:\n\
    > python train_classifier.py ../data/disaster_response_db.db classifier.pkl \n\
    Arguments Description: \n\
    1) Absolute path to SQLite destination database (e.g. DisasterResponse.db)\n\
    2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl")


if __name__ == '__main__':
    main()

