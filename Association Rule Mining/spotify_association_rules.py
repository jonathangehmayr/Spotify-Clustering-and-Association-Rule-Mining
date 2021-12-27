import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from time import time
from pprint import pprint
from typing import Tuple,List
import os


def load_data(n:int=1000)->pd.DataFrame:
    '''
    loads csv file and specifies how many samples should be randomly chosen
    '''
    os.chdir('..')
    df=pd.read_csv('Spotify_dataset.csv')
    os.chdir('./Association Rule Mining')
    df=df.sample(n=n, random_state=69)
    return df


def bin_continous_attributes(df:pd.DataFrame, attributes:list, bins:int=2)->pd.DataFrame:
    '''
    binning of continous features in equally spaced bins
    '''
    for attribute in attributes:
        # firstly binning the attribute to form nice labels
        _,bins_=pd.cut(df[attribute],bins=bins,retbins=True)
        bins_=list(pd.Series(bins_,dtype=float).round(3))#precision does not work in pd.cut
        # label creation with bin thresholds
        labels=['{} ({}, {})'.format(attribute,bins_[idx],bins_[idx+1])
            for idx,tresh in enumerate(bins_)
            if tresh!=bins_[-1]]
        #binning with the created labels
        df[attribute]=pd.cut(df[attribute],bins=bins, labels=labels)
    return df

def convert_categorial_attributes(df:pd.DataFrame, attributes:list)->pd.DataFrame:
    '''
    categorial features are not binned but just labelled differently 
    '''
    for attribute in attributes:
        df[attribute]=['{} ({})'.format(attribute,val) for val in df[attribute]]
    return df


def unicodeToAscii(s:str)->str:
    '''
    adapted from https://stackoverflow.com/a/518232/2809427
    normalizing text data to only contain ascii printables
    '''
    chars='''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in chars)

def custom_tokenize(s:str)->list:
    '''
    reuseable tokenizer using regextokenizer and stopwords from english and spanish language
    '''
    #https://www.rollingstone.com/music/music-features/english-speaking-artists-are-losing-their-grip-on-global-pop-domination-and-youtubes-leading-the-charge-786815/
    stop_words = list(stopwords.words('english'))+list(stopwords.words('spanish'))
    tokenizer = RegexpTokenizer(r"\w+")
    return [w for w in tokenizer.tokenize(s.lower()) if not w in stop_words]

def identity_tokenizer(text:str)->str:
    '''
    this tokenizer does not alter the text and is intended for the tfidfvectorizer to not 
    tokenize the text again after the text is tokenized before the tfidfvectorizer with
    the custom tokenizer
    '''
    return text


def tfidf_most_imp_words(ser_track:pd.Series, n:int=10)->list:
    '''
    inspired from: https://medium.com/kenlok/getting-the-top-words-in-a-multi-class-text-classification-problem-ad39e5a57eb2
    finding the most important words ranked with tfidf vectorizer and limited to n
    most important words
    '''
    tracks=[custom_tokenize(s) for s in ser_track]
    
    tfidfvectorizer=TfidfVectorizer(tokenizer=identity_tokenizer,lowercase=False)
    tfidf_wm = tfidfvectorizer.fit_transform(tracks)
    
    ser_most_imp_words = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidfvectorizer.get_feature_names()).sum().sort_values(ascending=False)  
    most_imp_words=list(ser_most_imp_words.iloc[:n].index)
    return most_imp_words

def bin_tracks_with_most_imp_words(ser_track:pd.Series,n_words:int=5)->list:
    '''
    if a track title contains a word that is in the most important words list
    it is saved for that sample, otherwise no words are stored
    '''
    ser_track=[unicodeToAscii(string) for string in ser_track]
    most_imp_words=tfidf_most_imp_words(ser_track,n=n_words)
    list_tracks=[]
    for s in ser_track:
        list_words=[]
        for word in custom_tokenize(s):
            if word in most_imp_words:
                if word not in list_words:
                    list_words.append('word ({})'.format(word))
        list_tracks.append(list_words)
    return list_tracks

def df_to_lists(df:pd.DataFrame, list_tracks:list)->List[list]:
    '''
    conversion of dataframe to list of lists and appending of track words list
    '''
    df_lists=df.values.tolist()
    for idx,lst in enumerate(df_lists):
        df_lists[idx]=lst+list_tracks[idx]
    return df_lists

def transform_item_list_to_binary(df_lists:List[list])->pd.DataFrame:
    '''
    transforming list of lists into binary dataframe
    '''
    te = TransactionEncoder()
    df_binary = te.fit(df_lists).transform(df_lists)
    df_binary = pd.DataFrame(df_binary, columns=te.columns_)
    return df_binary

def mine_association_rules(df_binary:pd.DataFrame)->pd.DataFrame:
    '''
    induction of association rules using apriori algorithm
    '''
    df_apriori = apriori(df_binary, min_support = 0.2, use_colnames = True, verbose = 1)
    print('Finished Apriori algorithm')
    df_ar = association_rules(df_apriori, metric = "confidence", min_threshold = 0.6)  
    print('Inferred association rules')
    df_ar_hits=df_ar[df_ar['consequents'] == frozenset({'hit (1)'})].sort_values('confidence', ascending=False).reset_index(drop=True)[:5]
    return df_ar_hits

def print_rules(df:pd.DataFrame)->None:
    df=df[['antecedents','support','confidence','lift']]
    df.loc[:,'antecedents']=[list(item) for item in df['antecedents']]
    indent='    '
    for item in df.iterrows():
        idx=item[0]
        list_ar=item[1][0]
        support=item[1][1]
        conf=item[1][2]
        lift=item[1][3]
        print('Association rules {}:'.format(idx))
        for ar in list_ar:
            print(indent+'{}'.format(ar))
        print()
        print(indent+'Support: {}'.format(round(support,ndigits=3)))
        print(indent+'Confidence: {}'.format(round(conf, ndigits=3)))
        print(indent+'Lift: {}'.format(round(lift,ndigits=3)))
        print()

if __name__ == '__main__':
    
    cont_attributes=['danceability','energy','loudness',
                'speechiness','acousticness','instrumentalness','valence',
                'liveness','tempo','duration_ms','chorus_hit','sections']
    
    cat_attributes=['key','mode','time_signature','decade','hit']
    
    df=load_data(n=1000)
    df=bin_continous_attributes(df, cont_attributes)
    df=convert_categorial_attributes(df, cat_attributes)
    ser_track=df['track']
    df=df.drop(['uri','track','artist'], axis='columns')
    
    list_tracks=bin_tracks_with_most_imp_words(ser_track,n_words=10)
    df_lists=df_to_lists(df, list_tracks)
    df_binary=transform_item_list_to_binary(df_lists)
    
    #induction of association rules requires some computation power and will take 1-3 min
    start=time()
    df_ar_hits=mine_association_rules(df_binary)
    stop=time()
    dur=stop-start
    print('Duration of computation: {}'.format(dur))
    print()
    
    print('Association rules for hit songs:')
    print()
    print_rules(df_ar_hits)
    print()