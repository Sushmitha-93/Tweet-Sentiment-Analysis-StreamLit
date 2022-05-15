import streamlit as st
import pandas as pd
import numpy as np
import snscrape.modules.twitter as sntwitter
import itertools

#Importing the pipeline from Transformers.
from transformers import pipeline
sentiment_classifier = pipeline('sentiment-analysis')

res=False

def run_sentiment_analysis_for_text(text): 
    if(text!=''):
        sentiment=sentiment_classifier(text)
        score=sentiment[0]['score']
        label= 'NEUTRAL' if score<0.6 else sentiment[0]['label']
        st.write('Sentiment in the text is: ', label)
        st.write('Score: ',score)

def run_sentiment_analysis_for_hashtag(hashtag,num):
    if(hashtag!=''):
        num = (2 if num=='' else num)
        df1 = pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper("#"+hashtag).get_items(),int(num)))[['content']]

        # Do sentiment analysis for each sentiment
        df2 = (df1.assign(sentiment = lambda x: x['content'].apply(lambda s: sentiment_classifier(s)))
        .assign(label = lambda x: x['sentiment'].apply(lambda s: (s[0]['label'])),score = lambda x: x['sentiment']
        .apply(lambda s: (s[0]['score']))))
        df2.loc[df2.query("score<0.6").index,'label']='NEUTRAL'
        return df2[['content','label','score']]


st.title('Tweet sentiment classification')


text = st.text_area('Enter the Tweet text to analyze', placeholder='Enter/Paste the tweet',height=100)
st.button('Submit',key=1, on_click=run_sentiment_analysis_for_text(text))



with st.form("my_form"):
    col1, col2 = st.columns([3, 1])

    with col1:
        hashtag = st.text_input('Enter the topic Ex: vaccine', placeholder='Enter topic / #hashtag')    

    with col2:
        num = st.text_input('Enter no. of tweets', value="2",placeholder='Enter no. of latest tweets',)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        result_df=run_sentiment_analysis_for_hashtag(hashtag,num)
        res=True


if(res==True):
    st.write(result_df.label.value_counts())
    st.caption('Hover on cell to read the tweet')
    st.write('Sentiment analysis of '+num+' latest tweets on the topic')
    st.dataframe(result_df,1000)
    
