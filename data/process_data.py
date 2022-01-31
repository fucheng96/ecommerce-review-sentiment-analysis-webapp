# Import libraries
import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(order_reviews_filepath,
              orders_filepath,
              order_items_filepath,
              products_filepath,
              product_name_eng_filepath,
              customers_filepath):

    """
    INPUT:
        order_reviews_filepath - File path to customers' reviews in CSV format
        orders_filepath - File path to all orders made in CSV format
        order_items_filepath - File path to items purchased within each order in CSV format
        products_filepath - File path to products sold by Olist in CSV format
        product_name_eng_filepath - File path to product category names
                                    tranlated to english messages in CSV format
        customers_filepath - File path to customers and its location in CSV format
    OUTPUT:
        df - Combined dataset containing the reviews and related key information
    """

    # Import customers' reviews and related key information
    review_df = pd.read_csv(order_reviews_filepath)
    order_df = pd.read_csv(orders_filepath)
    order_item_df = pd.read_csv(order_items_filepath)
    prod_df = pd.read_csv(products_filepath)
    prod_trans_df = pd.read_csv(product_name_eng_filepath)
    cust_df = pd.read_csv(customers_filepath)

    # Filter key columns
    order_df = order_df[['order_id',
                         'customer_id',
                         'order_delivered_customer_date',
                         'order_estimated_delivery_date']]
    order_item_df = order_item_df[['order_id', 'product_id']]
    prod_df = prod_df[['product_id', 'product_category_name']]
    cust_df = cust_df[['customer_id', 'customer_city', 'customer_state']]

    # List each dataset and its key ID for data merging
    df_list = [order_df, order_item_df, prod_df, prod_trans_df, cust_df]
    key_id_list = ['order_id',
                   'order_id',
                   'product_id',
                   'product_category_name',
                   'customer_id']

    # Set counter to loop through the list
    i = 0
    for df_ in df_list:

        # Get respective key id from df_key_id_list
        id_ = key_id_list[i]
        print(id_)

        # Remove duplicates before merging (left-join)
        print('Merge with df using ' + id_ + ':')
        print('\n')
        df_.drop_duplicates(subset=[id_], inplace=True)
        review_df = review_df.merge(df_, on=id_, how='left')

        # Update counter
        i += 1

    return review_df


def clean_data(df):

    """
    INPUT:
        df - Combined dataset containing customer reviews and other key columns
    OUTPUT:
        df_cleaned - Cleansed dataset with no duplicated and missing reviews
    """

    # Remove "\r\n"
    df['review_comment_message'] = df['review_comment_message'].replace(r'\s+|\\n', '', regex=True)

    # Remove missing reviews from review dataset
    df_cleaned = df[~df['review_comment_message'].isnull()]
    df_cleaned = df_cleaned[df_cleaned['review_comment_message'] != '']

    # Drop duplicates
    df_cleaned.drop_duplicates(subset=['review_id', 'review_comment_message'], inplace=True)

    return df_cleaned


def positive_review_label(df):

    """
    INPUT:
        df - dataset with no duplicated reviews

    Note:
    Labelling whether the review is positive or negative based on review score with:
    positive_review_ind = 1 if score is 4 or 5; else = 0
    This assumption is necessary for supervised learning to enable the model
    to learn if it's a positive review or not.
    Due to time constraint, each review_comment_message is not manually assessed
    to label it as positive or not.
    However, this assumption is reasonable as customers would tend to write
    positive reviews given a high score (4 or 5).
    """

    # Creating the response variable
    if df['review_score'] >= 4:
        return 1
    else:
        return 0

def save_data(df, database_filename):

    """
    INPUT:
        df - Cleansed dataset with no duplicated reviews,
             other created key features and labelled response variable
        database_filename - Name of database which should end with ".db"
    """

    # Create the SQL engine
    engine = create_engine('sqlite:///' + 'data/' + database_filename)

    # Output SQL database
    database_filename_ori = database_filename.replace('.db', '')
    df.to_sql(database_filename_ori, engine, index=False, if_exists='replace')


def main():

    # Setting the directories
    cd = os.getcwd() + '\\data'

    # Getting the respective filepaths
    order_reviews_filepath = cd + '\\olist_order_reviews_dataset.csv'
    orders_filepath = cd + '\\olist_orders_dataset.csv'
    order_items_filepath = cd + '\\olist_order_items_dataset.csv'
    products_filepath = cd + '\\olist_products_dataset.csv'
    product_name_eng_filepath = cd + '\\product_category_name_translation.csv'
    customers_filepath = cd + '\\olist_customers_dataset.csv'

    # Load the data
    print('Loading datasets...'
          .format(order_reviews_filepath,
                  orders_filepath,
                  order_items_filepath,
                  products_filepath,
                  product_name_eng_filepath,
                  customers_filepath))

    df = load_data(order_reviews_filepath,
                  orders_filepath,
                  order_items_filepath,
                  products_filepath,
                  product_name_eng_filepath,
                  customers_filepath)

    # Perform data cleaning
    print('Cleaning data...')

    # Remove duplicates & label positive sentiment
    df = clean_data(df)
    df['positive_review_ind'] = df.apply(positive_review_label, axis=1)
    df['review_message_length'] = df['review_comment_message'].apply(lambda x: len(x.split()))

    # Saving data into SQLite database
    database_filename = 'ecomm_por_cust_review.db'
    print('Saving data...\n    DATABASE: {}'.format(database_filename))
    save_data(df, database_filename)

    print('Cleaned data saved to database!')


if __name__ == '__main__':
    main()
