## import necessary libraries

import sys
from sqlalchemy import create_engine
from time import time

# python libraries to analyse data and multidimensional arrays
import pandas as pd
import numpy as np

# libraries for text processing
import re
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# libraries for model building
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
#from sklearn.externals import joblib
import joblib

nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    
    """
    This function loads data from a SQL database and separates the data into feature matrix and target veriables.
    
    Parameter
    -----------
        database_filepath : str 
                    path to the database
        
    Returns
    ----------
        X : matrix of shape (n_samples, n_features)
           the feature matrix
        Y : matrix of shape (n_samples, n_targets) 
           the target variable
        category_names : list 
           names of the different categories present in the target variable Y
    
    """
    
    engine = create_engine("sqlite:///"+database_filepath) # creates a connection to the database
    df = pd.read_sql('SELECT * FROM disaster_response_df', con=engine)
    
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    
    return X, Y, category_names    


def text_to_word(text):
    
    """
    A function to clean an input text. The steps followed for the text cleaning are :
     
     1. Normalization i.e. conversion to lower case and punctuation removal
     2. Tokenization 
     3. Stop words removal
     4. Lemmatization
    
    Parameter 
    -----------
      text : str 
        the input text to be cleaned
      
    Returns 
    ----------
      lemm_token_list : list
             a list of tokens obtained after cleaning the text
        
    """
    
    word_list =  word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    word_nostop_list = [word for word in word_list if word not in stopwords.words("english")]
    pos_dict = {"N":wordnet.NOUN, "J":wordnet.ADJ, "V":wordnet.VERB, "R":wordnet.ADV}
    
    lemm_token_list = set()
    for token,pos_tag in nltk.pos_tag(word_nostop_list):
        try:
            lemm_token_list.add(WordNetLemmatizer().lemmatize(token, pos=pos_dict[pos_tag[0]]))
        except:
            pass

    return list(lemm_token_list)

class FindTokenNumber(BaseEstimator, TransformerMixin):
    
    """
    A custom transformer to calculate the total number of tokens present after processing an input text.
    
    Attribute
    -----------
    cleaning_func : 
                a python function to clean the text data 
    
    Methods
    -----------
    text_length(text) : 
                calculate the total number of tokens corresponding to an input text after processing
    
    fit(X, y=None) : 
                fit the transformer according to the given data
    
    transform(X) : 
                transform an input text to the number of tokens present in it after processing
    
    """
    
    def __init__(self, cleaning_func):
        
        """
        Constructs all the necessary attributes for the FindTokenNumber object.
        
        Parameter
        ----------
        cleaning_func : 
                 a python function to clean the text data 
        
        """
        
        self.cleaning_func = cleaning_func
        
    def text_length(self, text):
        
        """
        This method calculates the total number of tokens corresponding to an input text after processing.
        
        Parameter
        ----------
            text : str
                an input raw text
                
        Returns
        --------
            n_token : int
                   the total number of tokens corresponding to an input text after processing
        """
        
        n_token = len(self.cleaning_func(text))
        return n_token
    
    def fit(self, X, y=None):
        
        """
        fit the transformer according to the given data.
        
        Parameters
        -----------
        X : iterable
          an iterable which yields strings 
          
        Returns
        ----------
        self
        
        """
        return self
    
    def transform(self, X):
        
        """
        transform an input text to the number of tokens present in it after processing.
        
        Parameter
        ----------
        X : iterable
          an iterable which yields strings 
          
        Returns
        ----------
        count_df : dataframe
          a datframe containing total number of tokens corresponding to a text after processing  
        
        """
        
        X_len = [self.text_length(x) for x in X]
        count_df = pd.DataFrame(X_len)
        
        return count_df
    

def build_model(clf, param_grid):
    
    """
    A function to build Pipeline and GridsearchCV objects. The pipeline object is built with the following steps:
    
    1. 'features' : a FeatureUnion object with the following steps running parallelly:
         a. 'tfidf' : convert a collection of raw text documents to a count matrix weighted by TF-IDF
         b. 'token_count' : count the number of tokens corrsponding to the text after cleaning
    2. 'classifier' : the model to be used for classification
    
    Parameters 
    ----------
    clf : a classifier object
       the classifier
         
    param_grid : dictionary
          keys of this dictionary are the parameters of the input classifier and values are lists of different values of these arameters over             which grid search will be performed.
      
    Returns 
    ---------
    grid_cv : a GridsearchCV object 
         the estimator is the pipeline object
    
    """
    
    
    pipe = Pipeline([
            ('features', FeatureUnion([
                ('tfidf', TfidfVectorizer(analyzer=text_to_word)),
                ('token_count', FindTokenNumber(text_to_word))
            ])),
            ('classifier', MultiOutputClassifier(clf))
           ])
    
    grid_cv = GridSearchCV(pipe, param_grid, cv=3, verbose=3)

    return grid_cv

def model_prediction(model, true_x, true_y, col_names):
    
    """
    This function makes predictions on a given dataset.
    
    Parameters
    ------------
    model : 
         the model to be used to make predictions
    
    true_x : matrix of shape (n_samples, n_features)
         the input matrix to make predictions on
         
    true_y : matrix of shape (n_samples, n_targets)
         the true target variable 
    
    col_names : list
         a list of categories present in the target variable
         
    Returns
    ----------
    
    pred_df : dataframe
          the dataframe containing the predictions for all target categories
    
    true_df : dataframe
          the dataframe containing true values of all the target categories
         
    """
    
    pred_model = model.predict(true_x)
    assert (pred_model.shape == true_y.shape), "shapes do not match!!"
    pred_df = pd.DataFrame(pred_model, columns=col_names)
    true_df = pd.DataFrame(true_y, columns=col_names)
    
    return pred_df, true_df


def model_evaluation(pred_df, true_df, col_names, eval_score, average, score_name):
    
    """
    A function to evaluate the performance of a model.
    
    Parameters
    ------------
    pred_df : dataframe
         the dataframe containing the predictions for all target categories
    
    true_df : dataframe
         the dataframe containing true values of all the target categories
         
    col_names : list
         a list of categories present in the target variable
         
    eval_score : an evaluation metric of the sklearn.metrics class     
         the score to evaluated to test the model's performance
         
    average : str
         the type of averaging performed on the data to calculate the score
    
    score_name : str
         the name of the evaluated score 
         
    
    Returns
    ---------
    score_df : dataframe
          a dataframe containing the evaluated score for each of the target category   
    
    """
    
    score_dict = {}
    
    for column in col_names:
        if column=='related': # the category with multiclass
            score = eval_score(y_true=true_df[column], y_pred=pred_df[column], average=average)
        else:
            score = eval_score(y_true=true_df[column], y_pred=pred_df[column])
        score_dict[column] = score
        
    score_df = pd.DataFrame.from_dict(score_dict, columns=[score_name], orient='index')
    
    return score_df


def save_model(model, model_filepath):
    
    """
    The function to save the model.
    
    Parameters
    ------------
    model :
         the model to be saved
         
    model_filepath : str
         the path to the saved model
    
    Returns
    ---------
    None
    
    """
    
    joblib.dump(model, model_filepath)


def main():
    """
    The function to load data, build the model, train the model on this data and finally make predictions on unseen data. 
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        param_dict = {'classifier__estimator__alpha': [0.1, 0.5, 1.0]}
        model = build_model(clf=MultinomialNB(), param_grid=param_dict)
        
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('The best parameter is...')
        print(model.best_params_)
        
        print('Making prediction on test set...')
        pred_df, true_df = model_prediction(model, X_test, Y_test, category_names)
        
        print('Evaluating model...')
        precision_df = model_evaluation(pred_df, true_df, category_names, precision_score, 'weighted', 'precision_score')
        recall_df = model_evaluation(pred_df, true_df, category_names, recall_score, 'weighted', 'recall_score')
        f1_df = model_evaluation(pred_df, true_df, category_names, f1_score, 'weighted', 'f1_score')
        
        # concatenate the three score dataframes
        score_df = pd.concat([precision_df, recall_df, f1_df], axis=1)
        print('the final scores for the test set are...')
        print(score_df)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

# this will only be executed when this module is run directly
if __name__ == '__main__': 
    main()
