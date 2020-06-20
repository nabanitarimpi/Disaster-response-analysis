# importing necessary libraries

import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """
    This function loads data from two csv files and merges them into a single pandas dataframe.
    
    Parameters:
        messages_filepath, categories_filepath (strings) : path to the csv files
        
    Return:
        merged_df : a pandas dataframe
    
    """
   
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    
    merged_df = messages_df.merge(categories_df, on='id', how='outer')
    
    return merged_df     


def clean_data(df):
    
    """
    This function takes in a dataframe as an input and cleans data in that dataframe for future use.
    
    Parameter:
        df : the input dataframe
         
    Return: 
        cleaned_df : the dataframe after data cleaning
         
    """
    
    categories = df['categories'].str.split(";", expand=True) 
    row = categories.iloc[0,:]
    categories.columns = [x.split("-")[0] for x in row.values]
    
    for column in categories.columns:
         categories[column] = categories[column].apply(lambda x:int(x.split("-")[-1]))

    df.drop('categories', axis=1, inplace=True)
    cleaned_df = pd.concat([df,categories], axis=1)
    cleaned_df.drop_duplicates(inplace=True)
    
    assert (cleaned_df[cleaned_df.duplicated()==True].shape[0] == 0), "oops! still some duplicates present in the data"
    
    return cleaned_df


def save_data(df, database_filepath, table_name):
    
    """
    The function to save the cleaned dataframe in a SQL database.
    
    Parameters:
        df : the dataframe which is to be saved in the database
        database_filepath (string) : the path to the database
        table_name (string) : the name of the dataframe in the database
        
    Return:
        This funtion returns nothing 
    
    """
    
    engine = create_engine("sqlite:///"+database_filepath)  # creates a connection to the database
    df.to_sql(table_name, engine, if_exists='replace', index=False)       
    
    

def main():
    
    """
    The main function to load data, clean it and finally save the data into a SQL database.
    
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        
        table_name = 'disaster_response_df'
        save_data(df, database_filepath, table_name=table_name)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__': # this will only be executed when this module is run directly
    main()