## Table of Contents

1. [Overview](#Overview)
2. [Installation](#Installation)
3. [Program Execution](#Program-Execution)
4. [Web App Screenshots](#Web-App-Screenshots)
5. [Acknowledgements](#Acknowledgements)

## Overview
The main objective of this project is to perform Sentiment Analysis on customer reviews on a Brazilian E-Commerce platform **[Olist](https://olist.com/pt-br/)** using Natural Language Processing ('NLP'), to determine if the reviews are positive or negative overall.

### Data
The dataset contains information of 100k orders from 2016 to 2018 such as order status, price, payment and freight performance to customer location, product attributes, etc. The focal point here would be the **reviews written by the customers**. Once the customer receives the product, or when the estimated delivery date is due (whether customer receives the product or not), the customer gets a satisfaction survey by email to describe the purchase experience and write down some comments.

### Limitation
Note that one limitation in this model is that the response variable of whether the review is positive or negative was determined based on review score with:
- Positive review if the review score is 4 or 5
- Negative review if the review score is 1, 2 or 3.

This assumption is necessary for supervised learning to enable the model to learn if it is a positive review or not. Due to time constraint, each review is not manually assessed to label it as positive or not. However, this assumption is reasonable as customers would tend to write positive review given a high score (4 or 5).

### Model Results
Multiple models were tested - Random Forest, AdaBoost, Gradient Boosting, Neural Network and Ensemble Stacking (Combination between Random Forest & AdaBoost with Logistic Regression as final estimator). From the findings:
- AdaBoost & Gradient Boosting takes a short time, but achieves slightly lower accuracies around 84%.
- Ensemble Stacking & Neural Network achieves good accuracy levels around 87%, but took significantly longer runtime.
- Random Forest not only achieves good accuracy levels around 87% on par with more complex models, it achieved that in a short time as AdaBoost & Gradient Boosting. Therefore, it is the chosen as the model in the Machine Learning ('ML') Pipeline (along with GridSearch for further model improvements).

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

2. Screenshots of dataset used to train the ML pipeline.

   ![Screenshot 2](https://github.com/fucheng96/ecommerce-review-sentiment-analysis-webapp/blob/main/screenshots/training-dataset-overview.PNG)

## Acknowledgements

Kudos to Olist for releasing a [public dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce). It contains multiple datasets which provide many exciting areas to explore apart from NLP such as sales prediction, customer segmentation etc.
