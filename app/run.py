from flask import Flask
from flask import render_template, request

import re
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd

import plotly, json
from plotly.graph_objs import Bar


app = Flask(__name__)

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

engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table('disaster_response_df', engine)

model = joblib.load("../models/classifier.pkl")

@app.route('/')
@app.route('/index')

def index():
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    graph_one = [Bar(
                   x = genre_names,
                   y = genre_counts
                )]
    
    layout_one = {
                  'title' : 'Distribution of message genres',
                  'xaxis' : {
                             'title' : 'Genre'
                            },
                  'yaxis' : {
                             'title' : 'Count'
                            }
    
                 }
    
    graphs = []
    
    graphs.append(dict(data=graph_one, layout=layout_one))
    
    ids = ['figure-{}'.format(i) for i, _ in enumerate(graphs)]
    graphJSON =  json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')

def go():
    
    query = request.args.get('query', '')
    
    classification_labels = model.predict([query])[0]
    classification_result = dict(zip(df.columns[4:], classification_labels))
    
    return render_template(
                          'go.html', 
                           query=query,
                           classification_result = classification_result
                           )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)
    

if __name__ == "__main__":
   main() 