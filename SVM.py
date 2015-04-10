import numpy as np
import random 

from sklearn.metrics import f1_score
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.metrics import confusion_matrix
from preprocessing import *
from itertools import product

from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    random.seed(42)
    
    # Parameter space
    Cs = np.logspace(2.5, 8, num=6)
    
    TFIDFs = [True, False]
    stop_wordss = [None, 'english']
    max_featuress = np.linspace(10, 5000, 5)
    ngram_ranges = [(1, 1), (1, 2)]
    
    parameter_space = product(TFIDFs, stop_wordss, max_featuress, ngram_ranges)
    
    for TFIDF, stop_words, max_features, ngram_range in parameter_space:
            
        # Loading dataset
        data_matrix, rating_matrix = read_data()
        N_tweets = rating_matrix.shape[0]
        
        # Transforming dataset using bag of words
        transformed_tweets = cleaned_bag_of_words_dataset(data_matrix, 
                                                          TFIDF=TFIDF, 
                                                          stop_words=stop_words, 
                                                          max_features=int(max_features), 
                                                          ngram_range=ngram_range)
        
        # majority voting rating
        majority_rating = majority_voting_ratings(rating_matrix)
        
        for C in Cs:
            rating_pred = []
            rating_real = []
            
            for train, val in KFold(N_tweets, n_folds=20):
                X_train, X_val, y_train, y_val = transformed_tweets[train], transformed_tweets[val], majority_rating[train], majority_rating[val]
                
                clf = svm.SVC(C=C)
                clf.fit(X_train, y_train)
                
                rating_pred = np.append(rating_pred, clf.predict(X_val))
                rating_real = np.append(rating_real, y_val)
            
            cm = confusion_matrix(rating_real, rating_pred)
            f1_score_macro = f1_score(rating_real, rating_pred, average='macro')
            
            print "C =", C, ", TFIDF =", TFIDF, ", stop_words = ", stop_words, ", max_features = ", max_features, ", ngram_range = ", ngram_range
            #print cm
            print f1_score_macro
            print ""