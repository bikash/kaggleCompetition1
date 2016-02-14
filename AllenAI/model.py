#!/usr/bin/env python

import csv
import argparse
import numpy as np


class Question(object):
    def __init__(self, id, question, answers, correct=None):
        self.id = id
        self.question = question
        self.answers = answers
        self.correct = correct

    @staticmethod
    def create_question(items):
        if len(items) == 7:  # has correct answer
            return Question(int(items[0]), items[1], items[3:], correct=items[2])
        else:
            return Question(int(items[0]), items[1], items[2:])


def train(filename):
    pass

        
def classify(filename):
    with open(filename) as f:
        data = []
        reader = csv.reader(f, delimiter = '\t')
        next(reader)           # skip header
        for row in reader:
            data.append(row)
            
        qs = [Question.create_question(d) for d in data]
        ids = [q.id for q in qs]
        
        labels = ('A', 'B', 'C', 'D')
        rand_guess = np.random.randint(4, size=len(qs))
        ans = zip(ids, [labels[a] for a in rand_guess])
        
        return ans


def output(ans, filename):
    with open(filename, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(('id', 'correctAnswer'))
        for a in ans:
            writer.writerow(a)


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('train', help='training set')
    #parser.add_argument('test', help='test set')
    #parser.add_argument('output', help='output filename')
    #args = parser.parse_args()
    args =['data/training_set.tsv', 'data/validation_set.tsv','data/sample_submission.csv']
    train('data/training_set.tsv')
    ans = classify('data/validation_set.tsv')
    output(ans, 'data/sample.csv')