# simple sentiment analysis using text blob
## DOESNT WORK NEED LSTM...

from textblob import TextBlob
import glob
import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
import matplotlib.pyplot as plt

def analyse_sentiment(FOLDER):
    # function to analyse the sentiment of tweets in specific folder
    files  = glob.glob(FOLDER + '/*.txt')
    sentiment_timeline = []
    # extract text  : 
    for f in files:
        text = open(f,'r').readlines()
        count = 0
        for tweet in text:
            blob = TextBlob(tweet)
            sentiment_timeline.append(blob.sentiment[0])
            print(tweet,blob.sentiment[0])
            count += 1
            if count > 20:
                exit()

    #cumulative timeline : 
    cum_sentiment = [i + i-1 for i in sentiment_timeline[1:]]

    # plot to check 
    plt.plot(sentiment_timeline)
    plt.plot(cum_sentiment)
    plt.show()
    #save
    


if __name__ == "__main__":
    analyse_sentiment('/home/lau/GIT/Complex Systems Stock Market/twitter_data/stocks')