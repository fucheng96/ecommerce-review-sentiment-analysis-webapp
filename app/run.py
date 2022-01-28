# Import libraries
import os
import re
import json
import plotly
import pandas as pd
import joblib
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin

# Import natural language toolkit libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])


app = Flask(__name__)

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


class GetReviewLength(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_text_length = pd.Series(X).apply(lambda x: len(x.split()))
        return pd.DataFrame(X_text_length)


# Setting directories
cd = os.getcwd()
database_filepath = cd.replace('webapp\\app', 'webapp\\data') + '\\ecomm_por_cust_review.db'

# Create SQL engine to import SQLite database
print(database_filepath)
engine = create_engine('sqlite:///' + database_filepath)
df = pd.read_sql_table('ecomm_por_cust_review', engine)

# Load model
model_filepath = cd.replace('webapp\\app', 'webapp\\models') + '\\sentiment_classifier.pkl'
model = joblib.load(model_filepath)

# Index webpage to display bar graphs and and receives user input text for model
@app.route('/')
@app.route('/index')

def index():

    # Extract data needed for visuals
    # Graph 1: Distribution of review text length
    review_length_counts = df.groupby('review_message_length').count()['review_id']
    review_text_length = list(review_length_counts.index)

    # Graph 2: Distribution of Response Categories
    review_score_counts = df.groupby('review_score').count()['review_id']
    review_score_counts = review_score_counts.sort_values(ascending=True)
    review_score = list(review_score_counts.index)

    # Create visuals using Plotly
    graphs = [
        # Graph 1: Distribution of Review Text Length
        {
            'data': [
                Bar(
                    x = review_text_length,
                    y = review_length_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Review Text Length',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Length of Review"
                }
            }
        },

        # Graph 2: Distribution of Review Scores
        {
            'data': [
                Bar(
                    x = review_score,
                    y = review_score_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Review Scores',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Review Scores"
                }
            }
        }
    ]

    # Encode plotly graphs in JSON format
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with respective plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# Web page that handles user query and displays model results
@app.route('/go')

def go():

    # Save user input in query
    query = request.args.get('query', '')

    # Use model to predict sentiment for query
    sentiment_label = model.predict([query])[0]

    # Mapping the labels and get its sentiments
    sentiments = {
        0: "Sentimento Negativo",
        1: "Sentimento Positivo"
    }
    sentiment_result = sentiments.get(sentiment_label)

    # This will render the go.html template file
    return render_template(
        'go.html',
        query = query,
        sentiment_result = sentiment_result
    )


# def main():
#     app.run(host='0.0.0.0', port=3001, debug=True)
#
#
# if __name__ == '__main__':
#     main()
