# simple sentiment analysis using text blob

from textblob import TextBlob
import glob

def analyse_sentiment(PATH):
    # function to analyse the sentiment of tweets in specific folder
    files  = glob.glob(PATH + '/*.txt')
    sentiment_timeline = []
    # extract text  : 
    for f in files:
        text = open(f,'r').readlines()
        for tweet in text:
            blob = TextBlob(tweet)
            sentiment_timeline.append(blob.sentiment)[0]
    
    #save
    


if __name__ == "__main__":
    analyse_sentiment('../twitter_data/stocks')