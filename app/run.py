# Import libraries
import re
import json
import plotly
import pandas as pd
import joblib
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

# Import natural language toolkit libraries
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])


app = Flask(__name__)

def tokenize(text):

    """
    INPUT:
        text - Text message that would need to be tokenized
    OUTPUT:
        clean_tokens - List of tokens extracted from the text message
    """

    # Detect & remove punctuations from the message
    detected_punctuations = re.findall('[^a-zA-Z0-9]', text)
    for punctuation in detected_punctuations:
        text = text.replace(punctuation, " ")

    # Tokenize the words
    tokens = word_tokenize(text)

    # Lemmanitizer to reduce words to its stems
    lemmatizer = WordNetLemmatizer()

    # Return list of normalized tokens reduced to its stems
    cleaned_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return cleaned_tokens


# Load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# Load model
model = joblib.load('models/classifier.pkl')


# Index webpage to display cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():

    # Extract data needed for visuals
    # Graph 1: Distribution of Message Genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Graph 2: Distribution of Response Categories
    n_response_cols = 36
    resp_cat_counts = df.iloc[:,-n_response_cols:].sum()
    resp_cat_counts = resp_cat_counts.sort_values(ascending=False)
    resp_cat_names = list(resp_cat_counts.index)

    # Create visuals using Plotly
    graphs = [
        # Graph 1: Distribution of Message Genres
        {
            'data': [
                Bar(
                    x = genre_names,
                    y = genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        # Graph 2: Distribution of Response Categories
        {
            'data': [
                Bar(
                    x = resp_cat_names,
                    y = resp_cat_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Response Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Response Category"
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

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html template file
    return render_template(
        'go.html',
        query = query,
        classification_result = classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
