
# text editor
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# Import Libraries
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.models import load_model
from keras import layers
import streamlit as st
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


# load model
model_load = keras.models.load_model("FINAL_LSTM.tf")



def run():

  st.write('## Tweets Sentiment!')
  st.markdown('***')

        # Membuat Form
  with st.form(key='form_parameters'):
        account = st.text_input('username', value='')
        tweet_1 = st.text_input('Tweet 1', value='')
        tweet_2 = st.text_input('Tweet 2', value='')
        tweet_3 = st.text_input('Tweet 3', value='')   

        st.markdown('***')
        
            
        submitted = st.form_submit_button('Predict Tweets Sentiment')
            

  data_inf = { 
            'account': account, 
            'tweet': [tweet_1,tweet_2,tweet_3 ]

        }

  data_inf = pd.DataFrame(data_inf)

  ##################################################### PREPROCESSED
  nltk.download('wordnet')
    # Create Lemmatizer and Stopwords
  stpwds_eng = list(set(stopwords.words('english')))
  stpwds_eng.append(['oh','s'])

    # lemmatization
  
    # Initialize the lemmatizer
  lemmatizer = WordNetLemmatizer()


    #Create Preprocessing Function
  def pre_process(text):
    
    text = text.lower() #to lowercase
      
    text = re.sub("@[A-Za-z0-9_]+", " ", text)# Remove Mention
      
    text = re.sub("#[A-Za-z0-9_]+", " ", text)# Remove Hashtag
      
    text = re.sub(r"\\n", " ",text)# Remove \n
    
    text = text.strip() # Remove Whitespace
      
    text = re.sub(r"http\S+", " ", text) # Remove Link
    text = re.sub(r"www.\S+", " ", text)
      
    text = re.sub("[^A-Za-z\s']", " ", text)# Remove symbols, emojis

    text = re.sub(r'\b\w{2}\b', '', text)# Remove words with 2 letters or less

    tokens = word_tokenize(text)# Tokenization

    text = ' '.join([word for word in tokens if word not in stpwds_eng])# Remove Stopwords
      
    text = lemmatizer.lemmatize(text)# Lemmatizing using WordLemmatizer
      
    return text

    ## Preprocessing
  data_inf['text_processed'] = data_inf['tweet'].apply(lambda x: pre_process(x))

    #Remove word with 2 letter or less
  words_2 = re.compile(r'\W*\b\w{1,2}\b')

  data_inf['text_processed']=data_inf['text_processed'].apply(lambda x: words_2.sub('', x) )

  

  if submitted:
    # Prediction
    tweet_prediction = model_load.predict(data_inf['text_processed'])

    tweet_prediction = np.where(tweet_prediction>= 0.5,1,0)

    predicted_sentiment=[]
    for i in tweet_prediction:
      if i == 0:
                  predicted_sentiment.append('Not Depresed')
      else :
                  predicted_sentiment.append('Depresed')
    data_inf['prediction_sentiment'] = predicted_sentiment

    st.dataframe(data_inf)




