import pandas as pd 
import numpy as np
from nltk.tokenize import TweetTokenizer
import re

def prep_tweets():
    ### IN PROGRESS ###
    #read trianing data : 
    f_name1 = "../tweet_training_data/train.csv"
    #f_name2 = "../tweet_training_data/test.csv"
    f_name3 = "../tweet_training_data/judge-1377884607_tweet_product_company.csv"
    tweet_df1 = pd.read_csv(f_name1,engine='python')
    #tweet_df2 = pd.read_csv(f_name2,engine='python')
    tweet_df3 = pd.read_csv(f_name3,engine = 'python')
    print(tweet_df3.head())

    # encoding for pos en negative to 2,4
    tweet_df1.loc[tweet_df1.Sentiment == 0,'Sentiment'] = 2
    tweet_df1.loc[tweet_df1.Sentiment == 1,'Sentiment'] = 4
    tweet_df3.loc[tweet_df3.EMOTION == 'Negative emotion','EMOTION'] = 2
    tweet_df3.loc[tweet_df3.EMOTION == 'Positive emotion','EMOTION'] = 4
    tweet_df3 = tweet_df3[['tweet_text','EMOTION']]
    # change datatypes of df3
    # tweet_df3['tweet_text'] = tweet_df3['tweet_text'].astype('|S80') 
    # tweet_df3['EMOTION'] = tweet_df3['EMOTION'].astype('int32') 
    # print(tweet_df3.dtypes)
    # print(tweet_df1.dtypes)
    # weird = tweet_df3.iloc[49]
    # print(weird)
    # print(re.sub(r'[^\x00-\x7F]+',' ', weird.tweet_text))
    # exit()
    ## use tweettokizer to make data little cleaner : 
    tknzr = TweetTokenizer(strip_handles=True,reduce_len=True)
    result = tweet_df1['SentimentText'].apply(lambda x : " ".join(tknzr.tokenize(x)))
    result3 = tweet_df3['tweet_text'].apply(lambda x : re.sub(r'[^\x00-\x7F]+',' ',x))
    result3 = result3.apply(lambda x : " ".join(tknzr.tokenize(x)))
    tweet_df1.SentimentText = result
    tweet_df3.tweet_text = result3
    print(tweet_df3.head())

prep_tweets()