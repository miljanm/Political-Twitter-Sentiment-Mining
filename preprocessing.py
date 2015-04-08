import csv
import numpy as np
from datetime import datetime
from pprint import pprint
from itertools import islice


__author__ = 'miljan'


def read_data():
    with open('./data/tweets.tsv', 'rU') as file1:
        # filter removes the rows starting with # (comments)
        file_reader = csv.reader(filter(lambda x: x[0] != '#', file1), delimiter='\t', dialect=csv.excel_tab)
        # skip header
        file_reader.next()

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


def process_timestamp(timestamp):
    return datetime.strptime(timestamp, '%m/%d/%y %H:%M')


if __name__ == '__main__':
    data_matrix, rating_matrix = read_data()