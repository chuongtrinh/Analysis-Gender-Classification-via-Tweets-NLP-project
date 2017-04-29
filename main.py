import pandas as pd
import numpy as np
import getopt
import math
import sys
import preprocessor as proc
import csv as writeCSV
import glove_twokenize as g_proc
#https://github.com/s/preprocessor


def preprocessing2(text):
        text = text.decode('ascii','ignore')
        proc.set_options(proc.OPT.URL, proc.OPT.MENTION, proc.OPT.HASHTAG)
        clean_ver = proc.tokenize(text).lower()
        return str(clean_ver)

def preprocessing(text):
        text = text.decode('ascii','ignore')
        return g_proc.tokenize(text)

def main():
        input_file = "data/gender-classifier-DFE.csv"
        csv_read = pd.read_csv(input_file) # encoding
        csv_read = csv_read.replace(np.nan, '',regex = True)
        header = csv_read.head()
        rows = csv_read.values.tolist()

        '''_unit_id	
        _golden	_unit_state	
        _trusted_judgments	
        _last_judgment_at	
        gender	
        gender:confidence	
        profile_yn	
        profile_yn:confidence	
        created	description	fav_number	
        gender_gold	link_color	name	
        profile_yn_gold	profileimage	
        retweet_count	
        sidebar_color	
        text	
        tweet_coord	
        tweet_count	
        tweet_created	tweet_id	tweet_location	user_timezone
        '''
        rows = list(filter(lambda x: x[6] ==1, rows))
        male_dataset = list(filter(lambda x: x[5] == "male", rows))
        female_dataset = list(filter(lambda x: x[5] == "female", rows))
        brand_dataset = list(filter(lambda x: x[5] == "brand", rows))

        a1 = open('processed_data/all_descriptions.csv','wb')
        a2 = open('processed_data/all_tweets.csv','wb')
        writer_all_desc = writeCSV.writer(a1)
        writer_all_tweets = writeCSV.writer(a2)
        
        # combine female descriptions into a single list of 'words'
        f1 = open('processed_data/female.descriptions.csv','wb')
        f2 = open('processed_data/female.tweets.csv','wb')
        writer_desc = writeCSV.writer(f1)
        writer_tweets =writeCSV.writer(f2)
        for row in female_dataset:
                clean_description = preprocessing(row[10])
                if len(clean_description) > 0:
                        writer_all_desc.writerow([row[14],clean_description, 'female'])
                        writer_desc.writerow([row[14], clean_description])
                clean_tweet = preprocessing(row[19])
                if len(clean_tweet) > 0:
                        writer_all_tweets.writerow([row[14],clean_tweet, 'female'])
                        writer_tweets.writerow([row[14],clean_tweet])

        f1.close()
        f2.close()

        f1 = open('processed_data/male.descriptions.csv','wb')
        f2 = open('processed_data/male.tweets.csv','wb')
        writer_desc = writeCSV.writer(f1)
        writer_tweets =writeCSV.writer(f2)
        for row in male_dataset:
                clean_description = preprocessing(row[10])
                if len(clean_description) > 0:
                        writer_all_desc.writerow([row[14],clean_description, 'male'])
                        writer_desc.writerow([row[14], clean_description])
                clean_tweet = preprocessing(row[19])
                if len(clean_tweet) > 0:
                        writer_all_tweets.writerow([row[14],clean_tweet, 'male'])
                        writer_tweets.writerow([row[14],clean_tweet])

        f1.close()
        f2.close()
                
        f1 = open('processed_data/brand.descriptions.csv','wb')
        f2 = open('processed_data/brand.tweets.csv','wb')
        writer_desc = writeCSV.writer(f1)
        writer_tweets =writeCSV.writer(f2)
        for row in brand_dataset:
                clean_description = preprocessing(row[10])
                if len(clean_description) > 0:
                        writer_all_desc.writerow([row[14],clean_description, 'brand'])
                        writer_desc.writerow([row[14], clean_description])
                clean_tweet = preprocessing(row[19])
                if len(clean_tweet) > 0:
                        writer_all_tweets.writerow([row[14],clean_tweet, 'brand'])
                        writer_tweets.writerow([row[14],clean_tweet])

        f1.close()
        f2.close()
        a1.close()
        a2.close()
        

if __name__=="__main__":
    main()
