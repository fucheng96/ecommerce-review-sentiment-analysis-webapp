# Import core libraries
import os
import sys
import re
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

# Import natural language toolkit libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

# Scikit learn libraries
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report


def load_data(database_filepath):

    """
    INPUT:
        database_filepath - File path to SQLite database which should end with ".db"
    OUTPUT:
        X - Dataset containing all the model features
        Y - Dataset containing all 36 indicator columns for each response category
    """

    # Create SQL engine to import SQLite database
    engine = create_engine('sqlite:///' + database_filepath)
    conn = engine.raw_connection()
    cur = conn.cursor()

    # Import data table from database
    database_filename = database_filepath.split('\\data\\')[1].replace('.db', '')
    sql_command = "SELECT * FROM " + database_filename
    df = pd.read_sql(sql_command, con=conn)
    conn.commit()
    conn.close()

    # Split dataset 'df' to features and target columns
    X = df['review_comment_message']
    y = df['positive_review_ind']

    return X, y


def tokenizer(text):

    """
    INPUT:
        text - Text message that would need to be tokenized
    OUTPUT:
        clean_tokens - List of tokens extracted from the text message
    """

    # Remove punctuations
    detected_punctuations = re.findall('[^a-zA-Z0-9]', text)
    for punctuation in detected_punctuations:
        text = text.replace(punctuation, ' ')

    # Remove words with single letters
    text = ' '.join([w for w in text.split() if len(w) > 1])

    # Tokenize the words
    tokens = word_tokenize(text)

    # Lemmanitizer to reduce words to its stems
    lemmatizer = WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    # Remove stopwords in Portugese
    por_stopwords = stopwords.words('portuguese')
    for st in por_stopwords:
        if st in clean_tokens:
            clean_tokens.remove(st)

    return clean_tokens


# Based on attributes observed earlier, longer comments tend to associate
# with negative sentiments (lower review score)
class GetReviewLength(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_text_length = pd.Series(X).apply(lambda x: len(x.split()))
        return pd.DataFrame(X_text_length)


def build_model():

    """
    OUTPUT:
        Machine Learning pipeline using Random Forest to process and classify
        customer reviews (in Portugese) on Olist ecommerce platform
    """

    # Machine learning pipeline using Random Forest
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenizer)),
                ('tfidf', TfidfTransformer())
            ])),

            ('review_length', GetReviewLength())
        ])),

        ('clf', RandomForestClassifier(n_jobs=-1, verbose=2)) # Use all processors
    ])

    # List down the parameters for Grid Search
    parameters = {
        'clf__n_estimators': [100, 200],
        'clf__min_samples_split': [2, 4]
    }

    return GridSearchCV(pipeline, param_grid=parameters, cv=4)


def evaluate_model(model, X_test, y_test):

    """
    INPUT:
        model - Trained Machine Learning pipeline
        X_test - Test features
        Y_test - Test labels
    OUTPUT:
        Metrics including overall accuracy, precision, recall, f1-score & support
    """

    # Predict on test data using trained model
    y_pred = model.predict(X_test)

    # Display metrics
    print(f'Overall accuracy: {np.round(100 * (y_pred == y_test).mean().mean(), 2)} %')
    print(classification_report(y_test.values, y_pred))


def save_model(model, model_filepath):

    """
    INPUT:
        model - Trained Machine Learning pipeline
        model_filepath - File path to where the model is saved which should end with ".pkl"
    OUTPUT:
        Model saved in .pkl format
    """

    # Save model
    pickle.dump(model, open(model_filepath, 'wb'))


def main():

    # Setting the directories
    cd = os.getcwd()
    database_filepath = cd.replace('models', 'data') + '\\ecomm_por_cust_review.db'
    model_filepath = cd + '\\sentiment_classifier.pkl'

    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, y = load_data(database_filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=23)

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, y_test)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model is saved in .pkl format!')


if __name__ == '__main__':
    main()
