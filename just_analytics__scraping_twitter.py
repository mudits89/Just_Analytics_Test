# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 22:43:18 2018

@author: mudit
"""
#access twitter 
import tweepy
from tweepy import OAuthHandler
import pandas as pd
from random import shuffle


lst_c_key = ['W2OCkMO2ybGFVU4s7BgXFk0MO', 'TDdDg7XUOccCaFS9NvEg6ADr1', 'YlyNuGXxkCIwTbTENShCVyAV4', 'nOtfkpBPDIMKlaGKtaiHdCOrb']
lst_c_secret = ['Hm5dGOZNCQ3SliwMiSMZTRo35k7ZFfQoGmvjpVkpUseuTGEjeJ', 'cFF44BcWwJm37Tqit2ATj1nJl2CaMm4Tah1OscpHov3nMDoDiQ', 'bj5dbh3RpsHohAxMuVN3zkjfNGrtZUbJlwYueWwqOeDfHaijCj', 'doiicH9HWbohehSmaqDcEatP26vFarWeVo8wXVHYQopZClWY1V']
lst_acc_tkn = ['16280903-EgpG7X9XQbmcajILrMAK7Tf6FOmqHEnNc8sCvZDbf', '16280903-SHcxm2NHJy1CpFYa1muJ7tdulN99Dlhql45Nj5LpY', '16280903-hNz3jkAo2PePYt42gF0MaJLurEd6VWV744X0N52DO', '1045687664412774401-VVcWQdpib32bcruusclbwhuaNMBgxb']
lst_acc_secret = ['GWrIwseUzVQwxJx2L1VfkF4gBs9hNu6Bq1UbtjBSvmhSG', 'ehoStGnleKxgLl6tKxrwUjgWTTJ3rKp5PWhCHmKF2NTTf', 'pCt3HJLZ0ia3dlGQqAfmkY6erKIFaxtvpeH6CXwy5e5SG', 'niqkWiiuzIUmW5kp6WmJEZPL1gmjDwtr8Q4mdBTed7eWP']


lst_c_key.reverse()
lst_c_secret.reverse()
lst_acc_tkn.reverse()
lst_acc_secret.reverse()


def extract_tweets(api, query, max_tweets=2500):
    searched_tweets = [status for status in tweepy.Cursor(api.search, q=query).items(max_tweets)]
    searched_tweets_text = [tweet.text for tweet in searched_tweets]
    return searched_tweets, searched_tweets_text


def store_lst_to_csv(inp_lst, file_name):    
    print("Starting writing list to csv")
    pd.DataFrame(inp_lst, columns=['tweets']).to_csv(file_name, header=True, index=False)
    print("Completed writing to " + str(file_name))


def store_data_local():
    query__AI = 'airindia'
    query__SQ = 'SingaporeAir'
    total_queries = [query__AI, query__SQ]
    lst_all_tweets=[]
    for og_pos in range(len(total_queries)):
        pos=og_pos%len(lst_c_key)
        consumer_key = lst_c_key[pos]
        consumer_secret = lst_c_secret[pos]
        access_token = lst_acc_tkn[pos]
        access_secret = lst_acc_secret[pos]     
        print(og_pos, pos)
        print(consumer_key)
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        api = tweepy.API(auth)
        """
        for status in tweepy.Cursor(api.home_timeline).items(10):
            # Process a single status
            print(status.text)
            
        for friend in tweepy.Cursor(api.friends).items():
            print(friend._json)
        """
        max_tweets_fin = 2500
        tweets, tweet_txt = extract_tweets(api=api, query=total_queries[og_pos], max_tweets=max_tweets_fin)
        pth="C:/Users/mudit/Desktop/Data Analytics/Just_Analytics_Test/Scraping_twitter/"
        store_lst_to_csv(inp_lst=tweet_txt, file_name=pth+total_queries[og_pos]+".csv")
        lst_all_tweets.append(tweet_txt)
        print("\n\nCompleted processing " + total_queries[og_pos])
    return lst_all_tweets

lst_tweets=store_data_local()

import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
import string

from textblob import TextBlob


punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via', 'https', '"', '...', 'i']
stop.extend(['airindia','flight','india','airindiain','air','airport','in...','thing','airline'])
stop.extend(["AirIndia", "airindiain", "RT", "Express flight", "JetAirways", "pilot", "RavinarIN", "https", 
             "co", ",","in...","amp",""])
stop = [term.lower() for term in stop]

lst_tweets_mod = []

wrd_list_AI = []
for tweet in lst_tweets[0]:
    words = word_tokenize(tweet)
    wrd_list_AI.extend([wrd.lower().strip() for wrd in words if wrd.lower().strip() not in stop])
  
wrd_list_SQ = []
for tweet in lst_tweets[1]:
    words = word_tokenize(tweet)
    wrd_list_SQ.extend([wrd.lower().strip() for wrd in words if wrd.lower().strip() not in stop])

lst_tweets_mod.append(wrd_list_AI)
lst_tweets_mod.append(wrd_list_SQ)

global_lst = []
#print(word_tokenize(lst_tweets[0][0]))
for i in range(len(lst_tweets)):
    search_word_lst= []
    for k in lst_tweets[i]:
        search_word_lst.extend(word_tokenize(k))
    search_word_lst = [term.lower() for term in search_word_lst if term.lower().strip() not in stop]
    fd = nltk.FreqDist(search_word_lst)
    fd.plot(30,cumulative=False)
    counts = Counter(search_word_lst)
    print(counts.most_common(20))
    global_lst.append(search_word_lst)
    print("\n\n")


#convert list to df
df1 = pd.DataFrame(lst_tweets[0],columns = ["tweets"])
df2 = pd.DataFrame(lst_tweets[1],columns = ["tweets"])

def sentiment_calc(text):
    try:
        return TextBlob(text).sentiment[0]
    except:
        return None

df1['sentiment'] = df1['tweets'].apply(sentiment_calc)
mean_sent_AI = df1["sentiment"].mean()
max_sent_AI = df1["sentiment"].max()
min_sent_AI = df1["sentiment"].min()


df2['sentiment'] = df2['tweets'].apply(sentiment_calc)
mean_sent_SIA = df2["sentiment"].mean()
max_sent_SIA = df2["sentiment"].max()
min_sent_SIA = df2["sentiment"].min()

import matplotlib.pyplot as plt
plt.boxplot(df1.sentiment)
plt.show()

plt.boxplot(df2.sentiment)
plt.show()

#Wordcount
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

df1_print = print("There are {} observations and {} features in this dataset. \n"
                  .format(df1.shape[0],df1.shape[1]))
df2_print = print("There are {} observations and {} features in this dataset. \n".
      format(df2.shape[0],df2.shape[1]))

#convert list to df
df3 = pd.DataFrame(lst_tweets_mod[0],columns = ["tweets"])
df4 = pd.DataFrame(lst_tweets_mod[1],columns = ["tweets"])


text1 = df3.tweets.str.cat(sep=', ')

wordcloud = WordCloud().generate(text1)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text1)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

text2 = df4.tweets.str.cat(sep=', ')

wordcloud = WordCloud().generate(text2)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text2)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
