### NOTEBOOK FOR GATHERING HISTORICAL TWEETS ###
import pandas as pd
from twitterscraper import query_tweets
import datetime as dt
import os
import pickle as pkl

class DBbuilder():
    def __init__(self):

        try: 
            self.load_class()
            print(self.resume_date)
            print(self.resume_querry)
        except:
            self.keywords = ['stocks','oil','forex','gold','commodities','crypto','CFD']
            self.help_querry = ' -filter:links' # min_replies:10 trading min_faves:10 
            self.begin_date = dt.date(2018,1,1)
            self.end_date = dt.date(2019,1,1)
            self.date_range = pd.bdate_range(self.begin_date,self.end_date)
            self.dt_list = self.date_range.strftime('%Y-%m-%d').tolist()
            self.home_folder = '/home/lau/GIT/Complex Systems Stock Market/twitter_data' 
            self.folder_list = []
            self.limit = 10000
            self.resume_querry = 3
            self.resume_date = 193
        
    def create_folders(self):
        # create folder of every keyword of different querries
        paths = []
        for key_word in self.keywords:
            path = self.home_folder + '/' + key_word
            try:
                os.mkdir(path)
                paths.append(path)
            except:
                print ("Creation of key directory %s failed" % path)
                
        # now in every folder create monthly folders : 
        # for path_ in paths:
        #     for date in self.dt_list:
        #         path = path_ + '/' + date
        #         try:
        #             os.mkdir(path)
        #             self.folder_list.append(path)
        #         except:
        #             print ("Creation of date directory %s failed" % path)
        
    def scrape(self):
        dates = list(zip(self.date_range[:-1],self.date_range[1:]))
        querries = list(map(lambda x: x + self.help_querry,self.keywords))
        for i,querry in enumerate(querries[self.resume_querry:],start=self.resume_querry):
            self.resume_querry = i
            for j,date in enumerate(dates[self.resume_date:],start=self.resume_date):
                #date_querry = ' since:' + date[0] + ' untill:' + date[1]
                tweets = query_tweets(querry,limit = self.limit, begindate = date[0].date(),
                                     enddate = date[1].date(), lang = 'english')
                self.resume_date = j
                fname = self.home_folder+'/'+self.keywords[i]+'/'+self.dt_list[j]+'.txt'
                file = open(fname,'w')
                for n,tweet in enumerate(tweets):
                    file.write(str(n) + ' ' + tweet.timestamp.strftime('%Y-%m-%d %H:%M:%S ') + tweet.text + '\n')
                file.close()
                # also save as pkl : 
                # pkl.dump(tweets,open(fname[:-3]+'.pkl','wb'))
                self.save_class()
            self.resume_date = 0
    
    def save_class(self):
        f = open('twitter_builder.pkl','wb')
        pkl.dump(self.__dict__, f)
        print(self.__dict__)

    def load_class(self):
        f = open('twitter_builder.pkl','rb')
        self.__dict__ = pkl.load(f)


if __name__ == "__main__":
    builder = DBbuilder()
    #builder.save_class()
    #builder.create_folders()
    builder.scrape()

    # def save(self):
    #     """save class as self.name.txt"""
    #     file = open(self.name+'.txt','w')
    #     file.write(cPickle.dumps(self.__dict__))
    #     file.close()

    # def load(self):
    #     """try load self.name.txt"""
    #     file = open(self.name+'.txt','r')
    #     dataPickle = file.read()
    #     file.close()

    #     self.__dict__ = cPickle.loads(dataPickle)
        
        
#             #Or save the retrieved tweets to file:
#     file = open(“output.txt”,”w”)
#     for tweet in query_tweets("Trump OR Clinton", 10):a
#         file.write(tweet.encode('utf-8'))
#     file.close()