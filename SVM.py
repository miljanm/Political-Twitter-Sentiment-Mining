import numpy as np
import random 

import gc

from sklearn.metrics import f1_score
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.metrics import confusion_matrix
from preprocessing import *
from itertools import product

from sklearn.feature_extraction.text import CountVectorizer
from enable.savage.svg.css.values import length

if __name__ == '__main__':
    random.seed(42)
    
    # Parameter space
    Cs = np.logspace(3, 8, num=5)
    
    stemmings = [True, False]
    TFIDFs = [False, True]
    stop_wordss = [None, 'english']
    max_featuress = np.linspace(10, 5000, 5).tolist() + [None]
    ngram_ranges = [(1, 1)] # ngram_ranges = [(1, 1),(1, 2)]
    lengths = [True, False]
    number_in_tweets = [True, False]
    words_presents = [['obama', 'mccain'], []]
    
    parameter_space = product(stemmings, TFIDFs, stop_wordss, max_featuress, ngram_ranges, lengths, number_in_tweets, words_presents)
    
    # Search the highest F1 macro score through parameter space 
    for stemming, TFIDF, stop_words, max_features, ngram_range, length, number_in_tweet, words_present in parameter_space:
            
        # Loading dataset
        data_matrix, rating_matrix = read_data()
        N_tweets = rating_matrix.shape[0]
        
        # Transforming dataset using bag of words
        transformed_tweets = cleaned_bag_of_words_dataset(data_matrix, 
                                                          stemming=stemming,
                                                          TFIDF=TFIDF, 
                                                          stop_words=stop_words, 
                                                          max_features=int(max_features), 
                                                          ngram_range=ngram_range, 
                                                          length=length, 
                                                          number_in_tweet=number_in_tweet, 
                                                          words_present=words_present)
        
        # majority voting rating
        majority_rating = majority_voting_ratings(rating_matrix)
        
        for C in Cs:
            rating_pred = []
            rating_real = []
            
            # Do Kfold cross validation
            fold = 0
            for train, val in KFold(N_tweets, n_folds=20):
                fold += 1
                #print fold 
                
                # Separate dataset into training and validation
                X_train, X_val, y_train, y_val = transformed_tweets[train], transformed_tweets[val], majority_rating[train], majority_rating[val]
                
                # Fit the SVM model
                clf = svm.SVC(C=C)
                clf.fit(X_train, y_train)
                
                # Predict the validation ratings
                rating_pred = np.append(rating_pred, clf.predict(X_val))
                rating_real = np.append(rating_real, y_val)
            
            # Calculate the confusion matrix and f1 macro score
            cm = confusion_matrix(rating_real, rating_pred)
            f1_score_macro = f1_score(rating_real, rating_pred, average='macro')
            
            # Print the results
            print "C =", C, ", stemming = ", stemming, ", TFIDF =", TFIDF, ", stop_words = ", stop_words, ", max_features = ", max_features, ", ngram_range = ", ngram_range, ", length = ", length, ", number_in_tweet = ", number_in_tweet, ", words_presents = ", words_presents
            #print cm
            print f1_score_macro
            print ""
            
            gc.collect()