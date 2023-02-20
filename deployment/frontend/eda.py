import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')



#melebarkan
st.set_page_config(
    page_title='Depressive Tweet?',
    #layout='wide',
    initial_sidebar_state='expanded'

)

st.markdown("""<style>.reportview-container {background: "5160549.jpg"}.sidebar .sidebar-content {background: "5160549.jpg"}</style>""",unsafe_allow_html=True)

nltk.download('wordnet')


def run():

    # Set title
    st.markdown("<h1 style='text-align: center; color: white;'>Depressive Tweet?</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey ;'></h3>", unsafe_allow_html=True)


    # library pillow buat gambar
    image = Image.open('depress.jpg')
    st.markdown('---')
    st.image(image, caption=' "But... " ') 

    # descripsi
    st.write('## Twitter Depression Sentiment ')

    # Membuat Garis lurus
    st.markdown('---')


    # Nampilin dataframe
    st.write('### Tweet Data Details')

    data = pd.read_csv('sentiment_tweets3.csv')
    st.dataframe(data.head(5))

    st.markdown('***')
    #barplot
    fig = plt.figure(figsize=(8,5))

    ###########################################

        # plot depression proportion
    st.write('### Depressing Tweet Proportion')

    st.write('There are lot more tweet with depressing tendency')

    # plotting data on chart

    fig_pie, ax = plt.subplots(figsize=(5,5))
    ax.pie(data=data ,x=data['label (depression result)'].value_counts(),labels=data['label (depression result)'].unique(), autopct='%.0f%%', explode = [0, 0.05], shadow=True, colors=sns.color_palette('pastel'))
    ax.set_title('Sentiment Proportion')
    st.pyplot(fig_pie)

    st.markdown('***')
    

    ############################################

    # Perbandingan 
    st.write('### Wordcloud Before and After Preprocessing ')
    st.write('#### Wordcloud Before Preprocessing ')

    # create wordclloud function
    def Plot_world(text_cloud):
        
        comment_words = ' '
        stopwords = set(STOPWORDS) 
        
        for val in text_cloud: 

            #each val to string 
            val = str(val) 

            # split
            tokens = val.split() 

            # lowercase 
            for i in range(len(tokens)): 
                tokens[i] = tokens[i].lower() 

            for words in tokens: 
                comment_words = comment_words + words + ' '

        wordcloud = WordCloud(width = 800, height = 800, 
                        background_color ='white', 
                        stopwords = stopwords, 
                        min_font_size = 10).generate(comment_words) 

        return wordcloud


    # cahnge columns for easier reading
    data = data.rename({'label (depression result)': 'label', 'message to examine':'message'}, axis=1)

    # plotting wordcloud
    text_cloud_0 = data[data.label==0].message.values
    text_cloud_1 = data[data.label==1].message.values

    fig_w, (ax1,ax2) = plt.subplots(1,2, figsize=(20,8))

    wordcloud_1 = Plot_world(text_cloud_1)
    wordcloud_0 = Plot_world(text_cloud_0)

    ax1.imshow(wordcloud_1)
    ax1.axis('off')
    ax1.set_title('Depressing Tweet Words\n')

    ax2.imshow(wordcloud_0)
    ax2.axis('off')
    ax2.set_title('Non Depressing Tweet Words\n')

    plt.tight_layout()

    st.pyplot(fig_w)

    ############################################################################################# WORDCLOUD AFTER
    #Create Preprocessing Function

    # Stopwords
    stpwds_eng = list(set(stopwords.words('english')))
    stpwds_eng.append(['oh','s'])

        # lemmatization
    nltk.download('wordnet')
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()


    def pre_process(text):
    
        text = text.lower() #to lowercase
        
        text = re.sub("@[A-Za-z0-9_]+", " ", text)# Remove Mention
        
        text = re.sub("#[A-Za-z0-9_]+", " ", text)# Remove Hashtag
        
        text = re.sub(r"\\n", " ",text)# Remove \n
        
        text = text.strip() # Remove Whitespace
        
        text = re.sub(r"http\S+", " ", text) # Remove Link
        text = re.sub(r"www.\S+", " ", text)
        
        text = re.sub("[^A-Za-z\s']", " ", text)# Remove symbols, emojis

        tokens = word_tokenize(text)# Tokenization

        text = ' '.join([word for word in tokens if word not in stpwds_eng])# Remove Stopwords
        
        text = lemmatizer.lemmatize(text)# Lemmatizing using WordLemmatizer
        
        return text
    # Applying the function of pre processing
    data['text_processed'] = data['message'].apply(lambda x: pre_process(x))


    #Remove word with 2 letter or less
    words_2 = re.compile(r'\W*\b\w{1,2}\b')

    data['text_processed'] = data['text_processed'].apply(lambda x: words_2.sub('', x) )

    st.write('#### Wordcloud After Preprocessing ')

    # create wordcloud
    # plotting wordcloud
    text_cloud_0_pr = data[data.label==0].text_processed.values
    text_cloud_1_pr = data[data.label==1].text_processed.values

    fig_wp, (ax1,ax2) = plt.subplots(1,2, figsize=(20,8))

    wordcloud_1_pr = Plot_world(text_cloud_1_pr)
    wordcloud_0_pr = Plot_world(text_cloud_0_pr)

    ax1.imshow(wordcloud_1_pr)
    ax1.axis('off')
    ax1.set_title('Depressing Tweet Words\n')

    ax2.imshow(wordcloud_0_pr)
    ax2.axis('off')
    ax2.set_title('Non Depressing Tweet Words\n')

    plt.tight_layout()

    st.pyplot(fig_wp)

    ####################################################

    st.markdown('***')



if __name__ == '__main__':
    run()