## Table of Contents

1. [Overview](#Overview)
2. [Installation](#Installation)
3. [Program Execution](#Program-Execution)
4. [Web App Screenshots](#Web-App-Screenshots)
5. [Acknowledgements](#Acknowledgements)

## Overview
The purpose of this project is to categorize messages received during a disaster to speed up rescue mission by respective organizations in terms of medical supplies, water supplies or fire fighting etc. 

This is part of Data Scientist Nanodegree Program by Udacity in collaboration with [Figure Eight aka Appen](https://appen.com/). The data were provided by Figure Eight and it contains tweet messages from real-life disasters and its respective labelled categories.

There are 3 main components to this project:
1. **ETL Pipeline**
   Extract data from given data source, transform the data through data wrangling and load them in a SQLite database.
   
2. **ML Pipeline**
   Train a machine learning model to classify the message categor(ies) given input message.
   
3. **Web App**
   Output message categories instantly in real-time using ML pipeline.

## Installation

1. Clone this git repository to your local workspace.
   
   `git clone https://github.com/fucheng96/disaster-response-webapp.git`
   
2. Install following dependencies in addition to the standard libraries from Anaconda distribution of Python.

    - Natural Language Process Libraries - [NLTK](https://www.nltk.org/)
    - SQLlite Database Libraries - [SQLalchemy](https://www.sqlalchemy.org/)
    - Web App using Python - [Flask](https://flask.palletsprojects.com/en/2.0.x/)
    - Data Visualization - [Plotly](https://plotly.github.io/plotly.py-docs/index.html)

## Program Execution
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in SQLite database.
      
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
      
    - To run ML pipeline that trains classifier using AdaBoost and saves in pickle format (.pkl).
      
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the project's root directory to run your web app.
   
   `python app/run.py`

3. Go to http://0.0.0.0:3001/ to view the web app.
   
   If you are using Windows and unable to access http://0.0.0.0:3001/, you may replace "0.0.0.0" with your local IPv4 address which should looks like "192.XXX.X.XXX". Actual link will then be http://192.XXX.X.XXX:3001/. To get IPv4 address, follow the steps below:
   
   a. Go to Command Prompt by typing `cmd` in the search bar.
   
   b. Type `ipconfig` and press enter.
   
   c. Look for "IPv4 Address" which looks like "192.XXX.X.XXX".

## Web App Screenshots

1. Screenshot of landing page. 

   ![Screenshot 1](https://github.com/fucheng96/disaster-response-webapp/blob/main/screenshots/main_page.jpeg?raw=true)

2. Screenshots of dataset used to train the ML pipeline.

   ![Screenshot 3](https://github.com/fucheng96/disaster-response-webapp/blob/main/screenshots/training_dataset_overview.PNG?raw=true)

## Acknowledgements

Kudos to Figure Eight for providing the labelled message data to ease model training, and the team behind the [Udacity Data Scientist Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025) for the code structure and materials for reference!
