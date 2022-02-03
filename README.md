## Table of Contents

1. [Overview](#Overview)
2. [Installation](#Installation)
3. [Folder Directory](#Folder-Directory)
4. [Program Execution](#Program-Execution)
5. [Web App Screenshots](#Web-App-Screenshots)
6. [Acknowledgements](#Acknowledgements)

## Overview

The main objective of this project is to perform Sentiment Analysis on customer reviews on a Brazilian E-Commerce platform **[Olist](https://olist.com/pt-br/)** using Natural Language Processing ('NLP'), to determine if the reviews are positive or negative.

The dataset contains information of 100k orders from 2016 to 2018 such as order status, price, payment and freight performance to customer location, product attributes, etc. The focal point here would be the **reviews written by the customers**. Once the customer receives the product, or when the estimated delivery date is due (whether customer receives the product or not), the customer gets a satisfaction survey by email to describe the purchase experience and write down some comments.

More details can be found in the [Jupyter Notebook here](https://github.com/fucheng96/ecommerce-review-sentiment-analysis-webapp/blob/main/review-sentiment-analysis-nlp.ipynb).

There are 3 main components to this project:
1. **ETL Pipeline**<br>
   Extract data from given data source, combine between multiple datasets, transform the data through data wrangling and load them in a SQLite database.

2. **ML Pipeline**<br>
   Train a machine learning model to classify the sentiment (positive or negative) given the Portugese (Brazilian) review message.

3. **Web App**<br>
   Output review or comment message sentiment instantly in real-time using ML pipeline.

## Installation

1. Clone this git repository to your local workspace.

   `git clone https://github.com/fucheng96/ecommerce-review-sentiment-analysis-webapp.git`

2. Install following dependencies in addition to the standard libraries from Anaconda distribution of Python.

    - Natural Language Process Libraries - [NLTK](https://www.nltk.org/)
    - SQLlite Database Libraries - [SQLalchemy](https://www.sqlalchemy.org/)
    - Web App using Python - [Flask](https://flask.palletsprojects.com/en/2.0.x/)
    - Data Visualization - [Plotly](https://plotly.github.io/plotly.py-docs/index.html)

3. Install [Git Large File Transfer ("LFS")](https://git-lfs.github.com/) to upload the 190MB-sized model (in .pkl format) to Github. Below are the steps taken:

    1. Install Git LFS in the Command Prompt using the command `git lfs install`.
    2. Head to directory of working folder and enter the command `git lfs track "*.pkl"` in Command Prompt to enable Git LFS to track files in ".pkl" format, which will create a text file named ".gitattributes.txt".
    3.  Be sure to track ".gitattributes.txt" file using `git add .gitattributes`, before the usual commit and push to  GitHub.

## Folder Directory

- [app](https://github.com/fucheng96/ecommerce-review-sentiment-analysis-webapp/tree/main/app)<br>
  Consists of Python file that launch the web app hosted locally and its related HTML files on the web layout and handling of queries.
   - templates
      - go.html
      - master.html
   - run.py
- [data](https://github.com/fucheng96/ecommerce-review-sentiment-analysis-webapp/tree/main/data)<br>
  Consists of Olist's public dataset downloaded from Kaggle, Python file that processes the raw data and saves it in SQLite database. 
   - olist_customers_dataset.csv
   - olist_order_items_dataset.csv
   - olist_order_reviews_dataset.csv
   - olist_orders_dataset.csv
   - olist_products_dataset.csv
   - product_category_name_translation.csv
   - process_data.py
   - ecomm_por_cust_review.db
- [models](https://github.com/fucheng96/ecommerce-review-sentiment-analysis-webapp/tree/main/models)<br>
  Consists of the Python file that specifies the text preparation and Machine Learning pipeline, the model saved in pickle format and results table that shows the comparison between various models explored in the [Jupyter Notebook here](https://github.com/fucheng96/ecommerce-review-sentiment-analysis-webapp/blob/main/review-sentiment-analysis-nlp.ipynb).
   - train_classifier.py
   - sentiment_classifier.pkl
   - model_results_table.xlsx
- [screenshots](https://github.com/fucheng96/ecommerce-review-sentiment-analysis-webapp/tree/main/screenshots)<br>
  Consists of the screenshots of the web app to be used in this README file.
   - main-page.PNG
   - sample-results.PNG
   - training-dataset-overview.PNG 
- [review-sentiment-analysis-nlp.ipynb](https://github.com/fucheng96/ecommerce-review-sentiment-analysis-webapp/blob/main/review-sentiment-analysis-nlp.ipynb)<br>
  
- [.gitattributes](https://github.com/fucheng96/ecommerce-review-sentiment-analysis-webapp/blob/main/.gitattributes)<br>

## Program Execution

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in SQLite database.

      `python data/process_data.py`

    - To run ML pipeline that trains classifier using Random Forest with Grid Search and saves in pickle format (.pkl).

      `python models/train_classifier.py`

2. Run the following command in the project's root directory to run your web app.

   `python app/run.py`

3. Go to http://0.0.0.0:3001/ to view the web app.

   If you are using Windows and unable to access http://0.0.0.0:3001/, you may replace "0.0.0.0" with your local IPv4 address which should looks like "192.XXX.X.XXX". Actual link will then be http://192.XXX.X.XXX:3001/. To get IPv4 address, follow the steps below:

   a. Go to Command Prompt by typing `cmd` in the search bar.

   b. Type `ipconfig` and press enter.

   c. Look for "IPv4 Address" which looks like "192.XXX.X.XXX".

## Web App Screenshots

1. Screenshot of landing page.

   ![Screenshot 1](https://github.com/fucheng96/ecommerce-review-sentiment-analysis-webapp/blob/main/screenshots/main-page.PNG)

2. Screenshot of overview of dataset used to train the ML pipeline.

   ![Screenshot 2](https://github.com/fucheng96/ecommerce-review-sentiment-analysis-webapp/blob/main/screenshots/training-dataset-overview.PNG)

3. Screenshot of ML pipeline query results.

   ![Screenshot 3](https://github.com/fucheng96/ecommerce-review-sentiment-analysis-webapp/blob/main/screenshots/sample-results.PNG)

## Acknowledgements

Kudos to Olist for releasing a [public dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce). It contains multiple datasets which provide many exciting areas to explore apart from NLP such as sales prediction, customer segmentation etc.
