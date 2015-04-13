import csv
import numpy as np
import string as str
from datetime import datetime
from pprint import pprint
from itertools import islice

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models.word2vec import Word2Vec


__author__ = 'miljan'


def read_data():
    with open('./data/tweets.tsv', 'rU') as file1:
        # filter removes the rows starting with # (comments)
        file_reader = csv.reader(filter(lambda x: x[0] in str.digits, file1), delimiter='\t', dialect=csv.excel_tab)

        # matrix containing all 8 ranking for each tweet
        rating_matrix = []
        data_matrix = []
        for row in file_reader:
            data_matrix.append([row[0]] + [process_timestamp(row[1])] + row[2:5])
            rating_matrix.append(row[5:])

        # first pad each array with 0 until the end, and then also replace any '' with 0
        rating_matrix = [[x if x is not '' else '0' for x in y + ['0']*(8 - len(y))] for y in rating_matrix]
        # convert to numpy ints
        rating_matrix = np.array(rating_matrix, dtype='int')

    return data_matrix, rating_matrix


def argmax(arr):
    max_element = max(arr)
    i = 0 
    arg_maxes = []
    for element in arr:
        if element == max_element:
             arg_maxes.append(i)
        i += 1
    
    # return the least confident result
    return arg_maxes[-1]

def process_timestamp(timestamp):
    return datetime.strptime(timestamp, '%m/%d/%y %H:%M')


def list_of_ints_from_string(s):
    l = []
    for t in word_tokenize(s.decode("utf8")):
        try:
            l.append(int(float(t)))
        except ValueError:
            pass
        
    return l


def cleaned_bag_of_words_dataset(data_matrix, stemming=False, stop_words=None, TFIDF=False, ngram_range=(1, 1), max_features=None,
                                 length=False, number_in_tweet=False, words_present=[]):
    if stemming:
        stemmer = SnowballStemmer("english")
        tweets = [" ".join([stemmer.stem(word) for word in word_tokenize(data_point[2].lower().decode("utf8"))]) for data_point in data_matrix]
    else:
        tweets = [data_point[2].lower() for data_point in data_matrix]
        
    if TFIDF:
        vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=ngram_range, max_features=max_features)
    else:
        vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range, max_features=max_features)
    
    dataset = vectorizer.fit_transform(tweets).toarray()
    
    if length:
        lengths = np.array([[len(word_tokenize(data_point[2].decode("utf8")))] for data_point in data_matrix])
        dataset = np.concatenate((dataset, lengths), axis=1)
     
    if number_in_tweet:
        numbers = []
        for data_point in data_matrix:
            number_list = list_of_ints_from_string(data_point[2])
            filtered_number_list = [number for number in number_list if abs(number) < 10]
            if len(filtered_number_list) == 0:
                numbers.append([0])
            else:
                numbers.append([np.mean(filtered_number_list)])
        dataset = np.concatenate((dataset, numbers), axis=1)

    for word in words_present:
        word_present = np.array([[int(word.lower() in word_tokenize(data_point[2].lower().decode("utf8")))] for data_point in data_matrix])
        dataset = np.concatenate((dataset, word_present), axis=1)
        
    return dataset


# derive sentence representation have sum of word vectors
def _build_sent_vec_as_sum(clean_sent, model):
    temp = np.zeros((1, 300))
    for word in clean_sent:
        try:
            temp += model[word]
        except:
            pass
    return temp


# derive sentence representation as average of word vectors
def _build_sent_vec_as_average(clean_sent, model):
    temp = np.zeros((1, 300))
    count = 0
    for word in clean_sent:
        try:
            temp += model[word]
            count += 1
        except:
            pass
    return temp/count if count > 0 else temp


def word2vec_features(data_matrix, stemming=False, stop_words=None, TFIDF=False, ngram_range=(1, 1), max_features=None,
                      length=False, number_in_tweet=False, words_present=[], policy='sum'):
    print '\n------------------'
    print 'Creating feature vector matrix...\n'
    if stemming:
        print '\n------------------'
        print 'Stemming...'
        stemmer = SnowballStemmer("english")
        tweets = [" ".join([stemmer.stem(word) for word in word_tokenize(data_point[2].lower().decode("utf8"))]) for data_point in data_matrix]
    else:
        tweets = [data_point[2].lower() for data_point in data_matrix]

    print '\n------------------'
    print 'Loading word2vec model...'

    model = Word2Vec.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)  # C binary format

    # determine the policy on how to build vectors
    if policy == 'sum':
        policy = _build_sent_vec_as_sum
    else:
        policy = _build_sent_vec_as_average

    print 'Applying word2vec model...'

    # create a len(tweets) x 300 dimensional matrix
    dataset = np.squeeze(np.array([policy(sent, model) for sent in tweets]))

    print "Done"

    if length:
        lengths = np.array([[len(word_tokenize(data_point[2].decode("utf8")))] for data_point in data_matrix])
        dataset = np.concatenate((dataset, lengths), axis=1)

    if number_in_tweet:
        numbers = []
        for data_point in data_matrix:
            number_list = list_of_ints_from_string(data_point[2])
            filtered_number_list = [number for number in number_list if abs(number) < 10]
            if len(filtered_number_list) == 0:
                numbers.append([0])
            else:
                numbers.append([np.mean(filtered_number_list)])
        dataset = np.concatenate((dataset, numbers), axis=1)

    for word in words_present:
        word_present = np.array([[int(word.lower() in word_tokenize(data_point[2].lower().decode("utf8")))] for data_point in data_matrix])
        dataset = np.concatenate((dataset, word_present), axis=1)

    print '\n------------------'
    print 'Feature vector constructed.'
    return dataset


def majority_voting_ratings(rating_matrix):
    majority_rating = []
    for ratings in rating_matrix:
        majority_rating.append(argmax(np.bincount(ratings)[1:]))
    
    return np.array(majority_rating)


def convert_4_to_3(x):
    if x == 4:
        return 3
    return x


def majority_voting_ratings_merge_3_4(rating_matrix):
    majority_rating = []
    for ratings in rating_matrix:
        ratings = map(convert_4_to_3, ratings)
        majority_rating.append(argmax(np.bincount(ratings)[1:]))
    
    return np.array(majority_rating)


if __name__ == '__main__':
    data_matrix, rating_matrix = read_data()