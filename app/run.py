# import packages from flask library
from flask import Flask
from flask import render_template, request

# import packages for text processing
import re
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# import packages for data loading and analysing
from sqlalchemy import create_engine
#from sklearn.externals import joblib
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# import packages for visualisation
import plotly, json
from plotly.graph_objs import Bar, Scatter
from wordcloud import WordCloud, STOPWORDS

#nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

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


def plotly_wordcloud(text, category):
    
    """
    A function to create visualisation of text data
    
    Parameters 
    -------------
    text : str 
        the input text to be used for word cloud
        
    category : str
        the category to which the input text belongs
      
    Returns 
    ----------
    graph : list
       a collection of data and specifications for visualisation
             
    layout : list    
       the plotly layout object
         
    """
    wc = WordCloud(stopwords=set(STOPWORDS), max_words=250, max_font_size=80)
    wc.generate(" ".join(text))
    
    word_list, freq_list, color_list, position_list = [], [], [], []
    

    for (word, freq), _, position, _, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        position_list.append(position)
        color_list.append(color)
        
    x =[i[0] for i in position_list]
    y =[i[1] for i in position_list]
    new_freq_list = [(i * 100) for i in freq_list]
    
    graph = [Scatter(
                     x = x,
                     y = y,
                     textfont = dict(size=new_freq_list,
                                     color=color_list),
                     hoverinfo='text',
                     hovertext=['{0}{1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                     mode='text',  
                     text=word_list
                )]
    
    layout = {
              'title' : "wordcloud for category "+category, 
              'xaxis' : {
                         'showgrid' : False,
                         'showticklabels' : False,
                         'zeroline' : False
                        },
              'yaxis' : {
                         'showgrid' : False,
                         'showticklabels' : False,
                         'zeroline' : False
                         },
              'width' : 500,
              'height' : 500
              }
    
    return graph, layout 

# load the data from the database
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table('disaster_response_df', engine)

# load the model
model = joblib.load("../models/classifier.pkl")

# index webpage displays data visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
    
    """
    A function to render the homepage and index webpage
    
    """
    #graph 1
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
                            }, 
                  'width' : 500,
                  'height': 500
                 }
    
    # graph 2
    msg_length = df.message.apply(len)
    genre_names = df.genre
    
    graph_two = [Bar(
                   x = genre_names,
                   y = msg_length
                )]
    
    layout_two = {
                  'title' : 'Distribution of message length as a function of genres',
                  'xaxis' : {
                             'title' : 'Genre'
                            },
                  'yaxis' : {
                             'title' : 'Message length'
                            }, 
                  'width' : 500,
                  'height': 500
                 }
    
    # graph 3
    text = df[df['genre']=='direct']['message']
    graph_three, layout_three = plotly_wordcloud(text, 'direct') 
    graph_four, layout_four = plotly_wordcloud(text, 'news') 
    graph_five, layout_five = plotly_wordcloud(text, 'social') 
    
    # store all the graph objects into one single list 
    graphs = []
    
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    graphs.append(dict(data=graph_three, layout=layout_three))
    graphs.append(dict(data=graph_four, layout=layout_four))
    graphs.append(dict(data=graph_five, layout=layout_five))    
    
    # encode plotly graphs in JSON
    ids = ['figure-{}'.format(i) for i, _ in enumerate(graphs)]
    graphJSON =  json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

 
# web page that handles user query and displays model results
@app.route('/go')

def go():
    
    """
    A function to render the query web page
    
    """
    # save user input in query
    query = request.args.get('query', '')
    
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_result = dict(zip(df.columns[4:], classification_labels))
    
    # This will render the go.html
    return render_template(
                          'go.html', 
                           query=query,
                           classification_result = classification_result
                           )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

# this will only be executed when this module is run directly
if __name__ == "__main__":
   main()     
