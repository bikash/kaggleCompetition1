#!/usr/bin/env python

import sys, os, lucene

from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.util import Version

import argparse
import pandas as pd
import utils
import numpy as np
from joblib import Parallel, delayed
import IndexFiles
INDEX_DIR = "IndexFiles.index"
DATA_DIR="./data/wiki_data/"



"""
method2: Combine the words in the question and the words in each answer 
(Q_A, Q_B, Q_C, Q_D) for querys to Lucene. the search result  with highest scoring 
documents is the answer:
cd ./lucene_sample/
python IndexFiles.py  ../../kaggle_allen-master/data/wiki_data/
python SearchFiles.py  -p ./lucene_sample/ -q ./train-val-data/dev/validation_set.tsv  -d doc.pickle     score:0.41625
"""
def run_method2(searcher, analyzer, qname='./data/validation_set.tsv'):

    
    
    
    result_file = open('validation_submission.csv', 'w')
    result_file.write('id,correctAnswer\n')


    raw_data = pd.read_csv(qname, '\t')
    total_data_len = len(raw_data)
    for i in range(total_data_len):
        query_str = raw_data['question'][i];
        
        query_answer_a = query_str + " " + raw_data['answerA'][i];
        query_answer_b = query_str + " " + raw_data['answerB'][i];
        query_answer_c = query_str + " " + raw_data['answerC'][i];
        query_answer_d = query_str + " " + raw_data['answerD'][i];
        
        querys = [query_answer_a, query_answer_b, query_answer_c, query_answer_d]
        
        score = []
        ind = 0
        for query in querys:
            query = utils.item2str(query) 
            query = QueryParser(Version.LUCENE_CURRENT, "contents",
                            analyzer).parse(query)
            scoreDocs = searcher.search(query, 1).scoreDocs
            #print len(scoreDocs)
            #print "%s total matching documents." % len(scoreDocs)
            #print "score ", scoreDocs[0].score
            score.append(scoreDocs[0].score)
        
        idx = np.argmax(np.asarray(score, dtype=np.float32))
        result_str = ['A', 'B', 'C', 'D']
        result_file.write("%s,%s\n" %(raw_data['id'][i], result_str[idx]))
        


    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--keywords', type=str, default='data/keywords.txt', help='keywords filename')
    parser.add_argument('--get_data', type=int, default= 0, help='flag to get wiki data for IR')
    parser.add_argument('--index', type=int, default= 0, help='flag to get wiki data for IR')
    parser.add_argument('--qname', type=str, default= 'data/validation_set.tsv', help='file name for validation')
    args = parser.parse_args()
    

    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print 'lucene', lucene.VERSION
    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))



    if args.get_data:
        utils.get_wiki_docs(args.keywords)
    if args.index:
        IndexFiles.IndexFiles(DATA_DIR, os.path.join(base_dir, INDEX_DIR),
            StandardAnalyzer(Version.LUCENE_CURRENT))

    directory = SimpleFSDirectory(File(os.path.join(base_dir, INDEX_DIR)))
    searcher = IndexSearcher(DirectoryReader.open(directory))
    analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
    #run(searcher, analyzer)
    run_method2(searcher, analyzer, args.qname)
    del searcher