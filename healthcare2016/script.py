# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
import pandas as p

from scipy.sparse import hstack, vstack, csc_matrix

from sklearn import linear_model, cross_validation
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.feature_extraction.text import *
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import KernelPCA

from nltk import word_tokenize, clean_html
from nltk.stem import SnowballStemmer

import re
import datetime
import json
import cPickle as pickle
import multiprocessing
from functools import partial
from mpldatacursor import datacursor
from StringIO import StringIO

# Set a memory limit.
import resource
resource.setrlimit(resource.RLIMIT_AS, (10100 * 1048576L, -1L))  # 10.1GiB.
SEED = 57846821  # 11617 57846821 4512971451315520


def load_data(tfidf_url, tfidf_title, tfidf_body, tfidf_html, tfidf_links, tfidf_htmlbody):
    """
        Loading of the raw data and tf-idf processing based on the input parameters. First the data is sought in
        the "dump/" folder, if not found, it processes it and dumps the result so that it should only be generated once.

        Input:  for each {url, title, body, html, links, htmlbody} a list
                [tfidf analyzer, ngram_range, minimum term frequency, maximum term frequency].

        Output: a dict with {'train_url', 'train_title', 'train_body','train_html', 'train_links',
                             'train_htmlbody','test_url', 'test_title', 'test_body', 'test_html',
                              'test_links', 'test_htmlbody', 'train_rest', 'test_rest', 'y'}.
    """
    print "\n---------------------------------------"
    print "Loading raw data ...\n"

    # Read the boilerplate (json) data.
    train_json = list(np.array(p.read_table('data/train.tsv'))[:, 2])
    test_json = list(np.array(p.read_table('data/test.tsv'))[:, 2])

    # Read labels from training.
    y = np.array(p.read_table('data/train.tsv'))[:, -1]

    # Read raw_html.
    train_html = []
    train_clean_html = []
    for urlid in np.array(p.read_table('data/train.tsv'))[:, 1]:
        try:
            html = open('data/raw_content/' + str(urlid) + '-utf').read()
        except:
            html = open('data/raw_content/' + str(urlid) + '-iso').read()

        train_html.append(html.replace("\n", " ").replace("'", "").replace("&rsquo;", "").replace("&ldquo;", "").replace("&ndash;", ""))
        train_clean_html.append(clean_html(html).replace("\n", " ").replace("'", "").replace("&rsquo;", "").replace("&ldquo;", "").replace("&ndash;", ""))

    test_html = []
    test_clean_html = []
    for urlid in np.array(p.read_table('data/test.tsv'))[:, 1]:
        try:
            html = open('data/raw_content/' + str(urlid) + '-utf').read()
        except:
            html = open('data/raw_content/' + str(urlid) + '-iso').read()

        test_html.append(html.replace("\n", " ").replace("'", "").replace("&rsquo;", "").replace("&ldquo;", "").replace("&ndash;", ""))
        test_clean_html.append(clean_html(html).replace("\n", " ").replace("'", "").replace("&rsquo;", "").replace("&ldquo;", "").replace("&ndash;", ""))

    train_json_url = []
    train_json_title = []
    train_json_body = []

    # First interpreting json and splitting in {body, url, title} for training data.
    print " Splitting json training ..."
    for json_raw in train_json:
        io = StringIO(json_raw)
        json_data = json.load(io)
        if json_data.get('url') is not None:
            train_json_url.append(json_data.get('url').replace("'", ""))
        else:
            train_json_url.append(u'url_missing')
        if json_data.get('title') is not None:
            train_json_title.append(json_data.get('title').replace("'", ""))
        else:
            train_json_title.append(u'title_missing')
        if json_data.get('body') is not None:
            train_json_body.append(json_data.get('body').replace("'", ""))
        else:
            train_json_body.append(u'body_missing')

    test_json_url = []
    test_json_title = []
    test_json_body = []

    # Now for test data.
    print " Splitting json test ...\n"
    for json_raw in test_json:
        io = StringIO(json_raw)
        json_data = json.load(io)
        if json_data.get('url') is not None:
            test_json_url.append(json_data.get('url').replace("'", ""))
        else:
            test_json_url.append(u'url_missing')
        if json_data.get('title') is not None:
            test_json_title.append(json_data.get('title').replace("'", ""))
        else:
            test_json_title.append(u'title_missing')
        if json_data.get('body') is not None:
            test_json_body.append(json_data.get('body').replace("'", ""))
        else:
            test_json_body.append(u'body_missing')

    # Pre-process non-text features.
    print "---------------------------------------"
    print "Preprocessing non-text ...\n"
    try:
        train_rest = pickle.load(open("dump/train_rest.pkl", "rb"))
        test_rest = pickle.load(open("dump/test_rest.pkl", "rb"))
    except:
        # Could not read from dump, process again and dump.
        # Read other features.
        train_rest = p.read_table('data/train.tsv', na_values=['?'])
        test_rest = p.read_table('data/test.tsv', na_values=['?'])

        with open("dump/train_rest.pkl", "wb") as f:
                pickle.dump(train_rest, f, pickle.HIGHEST_PROTOCOL)
        with open("dump/test_rest.pkl", "wb") as f:
                pickle.dump(test_rest, f, pickle.HIGHEST_PROTOCOL)
        print "Dumped."

    # Pre-process text.
    print "\n---------------------------------------"
    print "Preprocessing text (TF-IDF) ... \n"

    class SnowballTokenizer(object):
            def __init__(self):
                self.wnl = SnowballStemmer('english')
            def __call__(self, doc):
                return [self.wnl.stem(t) for t in word_tokenize(" ".join(re.findall(r'\w+',
                                                                                    doc,
                                                                                    flags=re.UNICODE | re.LOCALE)))]

    stopwords_uni = ['i', 'http', 'www']

    tfidf = TfidfVectorizer(strip_accents='unicode', analyzer='word', tokenizer=SnowballTokenizer(),
                            lowercase=True, norm='l2', ngram_range=(1, 2), sublinear_tf=True,
                            use_idf=True, smooth_idf=True, max_features=None,
                            stop_words=stopwords_uni)
    # TF-IDF
    tfidf_stemmer = 'snowball'

    #########################################################################################
    #                                       URL
    #

    tfidf.set_params(analyzer=tfidf_url[0], ngram_range=tfidf_url[1], min_df=tfidf_url[2], max_df=tfidf_url[3])
    print(" Fitting TfidfVectorizer for url with params %s" % tfidf_url)

    # First try to read from cache/dump.
    try:
        X_train_url = pickle.load(open("dump/tf_idf_TRAIN_URL_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                                       % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                                          tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                                          tfidf_stemmer), "rb"))
        X_test_url = pickle.load(open("dump/tf_idf_TEST_URL_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                                      % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                                         tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                                         tfidf_stemmer), "rb"))
    except:
        # If it fails, then calculate again.
        print "\t Could not read from dump, fitting ..."
        tfidf.fit(train_json_url + test_json_url)

        # Transform and dump.
        X_train_url = tfidf.transform(train_json_url)
        with open("dump/tf_idf_TRAIN_URL_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(X_train_url, f, pickle.HIGHEST_PROTOCOL)

        X_test_url = tfidf.transform(test_json_url)
        with open("dump/tf_idf_TEST_URL_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(X_test_url, f, pickle.HIGHEST_PROTOCOL)
        with open("dump/tf_idf_FEATURES_URL_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(tfidf.get_feature_names(), f, pickle.HIGHEST_PROTOCOL)
        print "\t Dumped."

    print "\t Number of features: %i." % X_train_url.shape[1]

    #########################################################################################
    #                                       TITLE
    #

    tfidf.set_params(analyzer=tfidf_title[0], ngram_range=tfidf_title[1], min_df=tfidf_title[2], max_df=tfidf_title[3])
    print("\n Fitting TfidfVectorizer for title with params %s" % tfidf_title)

    # First try to read from cache/dump.
    try:
        X_train_title = pickle.load(open("dump/tf_idf_TRAIN_TITLE_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                                         % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                                            tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                                            tfidf_stemmer), "rb"))
        X_test_title = pickle.load(open("dump/tf_idf_TEST_TITLE_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                                        % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                                           tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                                           tfidf_stemmer), "rb"))
    except:
        # If it fails, then calculate again.
        print "\t Could not read from dump, fitting ..."
        tfidf.fit(train_json_title + test_json_title)

        # Transform and dump.
        X_train_title = tfidf.transform(train_json_title)
        with open("dump/tf_idf_TRAIN_TITLE_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(X_train_title, f, pickle.HIGHEST_PROTOCOL)

        X_test_title = tfidf.transform(test_json_title)
        with open("dump/tf_idf_TEST_TITLE_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(X_test_title, f, pickle.HIGHEST_PROTOCOL)
        with open("dump/tf_idf_FEATURES_TITLE_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(tfidf.get_feature_names(), f, pickle.HIGHEST_PROTOCOL)
        print "\t Dumped."

    print "\t Number of features: %i." % X_train_title.shape[1]

    #########################################################################################
    #                                       BODY
    #

    tfidf.set_params(analyzer=tfidf_body[0], ngram_range=tfidf_body[1], min_df=tfidf_body[2], max_df=tfidf_body[3])
    print("\n Fitting TfidfVectorizer for body with params %s" % tfidf_body)
    try:
        X_train_body = pickle.load(open("dump/tf_idf_TRAIN_BODY_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                                        % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                                           tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                                           tfidf_stemmer), "rb"))
        X_test_body = pickle.load(open("dump/tf_idf_TEST_BODY_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                                       % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                                          tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                                          tfidf_stemmer), "rb"))
    except:
        # If it fails, then calculate again.
        print "\t Could not read from dump, fitting ..."
        tfidf.fit(train_json_body + test_json_body)

        # Transform and dump.
        X_train_body = tfidf.transform(train_json_body)
        with open("dump/tf_idf_TRAIN_BODY_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(X_train_body, f, pickle.HIGHEST_PROTOCOL)

        X_test_body = tfidf.transform(test_json_body)
        with open("dump/tf_idf_TEST_BODY_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(X_test_body, f, pickle.HIGHEST_PROTOCOL)
        with open("dump/tf_idf_FEATURES_BODY_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(tfidf.get_feature_names(), f, pickle.HIGHEST_PROTOCOL)
        print "\t Dumped."

    print "\t Number of features: %i." % X_train_body.shape[1]

    #########################################################################################
    #                                       LINKS
    #

    class LinksTokenizer(object):
            def __init__(self):
                self.wnl = SnowballStemmer('english')
            def __call__(self, doc):
                return [self.wnl.stem(t) for t in word_tokenize(" ".join(re.findall(r'\[[^\]]*\]\([^\)]*\)',
                                                                                    doc,
                                                                                    flags=re.UNICODE | re.LOCALE)))]

    tfidf = TfidfVectorizer(strip_accents='unicode', analyzer='word', tokenizer=LinksTokenizer(),
                            lowercase=True, norm='l2', ngram_range=(1, 2), sublinear_tf=True,
                            use_idf=True, smooth_idf=True, max_features=None,
                            stop_words=stopwords_uni)

    tfidf.set_params(analyzer=tfidf_links[0], ngram_range=tfidf_links[1], min_df=tfidf_links[2], max_df=tfidf_links[3])
    print("\n Fitting TfidfVectorizer for links with params %s" % tfidf_links)

    try:
        X_train_links = pickle.load(open("dump/tf_idf_TRAIN_LINKS_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                                          % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                                             tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                                             tfidf_stemmer), "rb"))
        X_test_links = pickle.load(open("dump/tf_idf_TEST_LINKS_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                                        % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                                           tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                                           tfidf_stemmer), "rb"))
    except:
        # If it fails, then calculate again.
        print "\t Could not read from dump, fitting ..."
        tfidf.fit(train_clean_html + test_clean_html)

        # Transform and dump.
        X_train_links = tfidf.transform(train_clean_html)
        with open("dump/tf_idf_TRAIN_LINKS_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(X_train_links, f, pickle.HIGHEST_PROTOCOL)

        X_test_links = tfidf.transform(test_clean_html)
        with open("dump/tf_idf_TEST_LINKS_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(X_test_links, f, pickle.HIGHEST_PROTOCOL)
        with open("dump/tf_idf_FEATURES_LINKS_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(tfidf.get_feature_names(), f, pickle.HIGHEST_PROTOCOL)
        print "\t Dumped."

    print "\t Number of features: %i." % X_train_links.shape[1]

    #########################################################################################
    #                                       HTML
    #

    class HTMLTokenizer(object):
            def __init__(self):
                self.wnl = SnowballStemmer('english')
            def __call__(self, doc):
                # Remove some special chars (get removed by word_tokenize too).
                repl = re.sub(r'\#\#+', ' ', doc, flags=re.UNICODE | re.LOCALE)
                repl = re.sub(r'\#\s', ' ', repl, flags=re.UNICODE | re.LOCALE)
                # Remove the links from the html.
                repl = re.sub(r'\[[^\]]*\]\([^\)]*\)', ' ', repl, flags=re.UNICODE | re.LOCALE)
                return [self.wnl.stem(t) for t in word_tokenize(" ".join(re.findall(r'\w+',
                                                                                    repl,
                                                                                    flags=re.UNICODE | re.LOCALE)))]

    tfidf = TfidfVectorizer(strip_accents='unicode', analyzer='word', tokenizer=HTMLTokenizer(),
                            lowercase=True, norm='l2', ngram_range=(1, 2), sublinear_tf=True,
                            use_idf=True, smooth_idf=True, max_features=None,
                            stop_words=stopwords_uni)

    tfidf.set_params(analyzer=tfidf_html[0], ngram_range=tfidf_html[1], min_df=tfidf_html[2], max_df=tfidf_html[3])
    print("\n Fitting TfidfVectorizer for html with params %s" % tfidf_html)

    try:
        X_train_html = pickle.load(open("dump/tf_idf_TRAIN_HTML_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                                        % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                                           tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                                           tfidf_stemmer), "rb"))
        X_test_html = pickle.load(open("dump/tf_idf_TEST_HTML_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                                       % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                                          tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                                          tfidf_stemmer), "rb"))
    except:
        # If it fails, then calculate again.
        print "\t Could not read from dump, fitting ..."
        tfidf.fit(train_clean_html + test_clean_html)

        # Transform and dump.
        X_train_html = tfidf.transform(train_clean_html)
        with open("dump/tf_idf_TRAIN_HTML_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(X_train_html, f, pickle.HIGHEST_PROTOCOL)

        X_test_html = tfidf.transform(test_clean_html)
        with open("dump/tf_idf_TEST_HTML_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(X_test_html, f, pickle.HIGHEST_PROTOCOL)
        with open("dump/tf_idf_FEATURES_HTML_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(tfidf.get_feature_names(), f, pickle.HIGHEST_PROTOCOL)
        print "\t Dumped."

    print "\t Number of features: %i." % X_train_html.shape[1]

    #########################################################################################
    #                              HTML + BODY
    #

    tfidf.set_params(analyzer=tfidf_htmlbody[0], ngram_range=tfidf_htmlbody[1], min_df=tfidf_htmlbody[2], max_df=tfidf_htmlbody[3])
    print("\n Fitting TfidfVectorizer for htmlbody with params %s" % tfidf_htmlbody)

    try:
        X_train_htmlbody = pickle.load(open("dump/tf_idf_TRAIN_HTMLBODY_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                                            % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                                               tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                                               tfidf_stemmer), "rb"))
        X_test_htmlbody = pickle.load(open("dump/tf_idf_TEST_HTMLBODY_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                                           % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                                              tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                                              tfidf_stemmer), "rb"))
    except:
        # If it fails, then calculate again.
        print "\t Could not read from dump, fitting ..."
        concat_train = [''.join((train_clean_html[i], train_json_body[i].encode('unicode-escape'))) for i in range(len(train_clean_html))]
        concat_test = [''.join((test_clean_html[i], test_json_body[i].encode('unicode-escape'))) for i in range(len(test_clean_html))]

        tfidf.fit(concat_train + concat_test)

        #Transform and dump.
        X_train_htmlbody = tfidf.transform(concat_train)
        with open("dump/tf_idf_TRAIN_HTMLBODY_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(X_train_htmlbody, f, pickle.HIGHEST_PROTOCOL)

        X_test_htmlbody = tfidf.transform(concat_test)

        with open("dump/tf_idf_TEST_HTMLBODY_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(X_test_htmlbody, f, pickle.HIGHEST_PROTOCOL)
        with open("dump/tf_idf_FEATURES_HTMLBODY_analyzer_%s_range_%s_mindf_%f_maxdf_%f_%s.pkl"
                  % (tfidf.get_params()['analyzer'], tfidf.get_params()['ngram_range'],
                     tfidf.get_params()['min_df'], tfidf.get_params()['max_df'],
                     tfidf_stemmer), "wb") as f:
            pickle.dump(tfidf.get_feature_names(), f, pickle.HIGHEST_PROTOCOL)
        print "\t Dumped."

    print "\t Number of features: %i.\n" % X_train_htmlbody.shape[1]

    return {'train_url': X_train_url, 'train_title': X_train_title, 'train_body': X_train_body,
            'train_html': X_train_html, 'train_links': X_train_links, 'train_htmlbody': X_train_htmlbody,
            'test_url': X_test_url, 'test_title': X_test_title, 'test_body': X_test_body,
            'test_html': X_test_html, 'test_links': X_test_links, 'test_htmlbody': X_test_htmlbody,
            'train_rest': train_rest, 'test_rest': test_rest, 'y': y}


# Model functions.
def model_Chi2LR(x_train, y_train, train_predict, x_predict, params):
    """
        LR after chi2 feature selection.

        Input:  x_train = training_set,
                y_train = labels for the training set,
                train_predict = the first set to be predicted,
                x_predict = the second set to be predicted,
                params, the parameters for the model:
                    [lr_penalty, lr_C, lr_class_weight, gridsearch_bool, gridsearch_params,, chi2_perc],
                    chi2_perc is the percentile parameter for chi2 selection.

        Output: [prediction for first set, prediction for second set].
    """

    y_train = np.asarray(y_train, dtype='int')
    select = SelectPercentile(chi2, percentile=params[5])
    x_train = select.fit_transform(x_train, y_train)
    train_predict = select.transform(train_predict)
    x_predict = select.transform(x_predict)

    model = linear_model.LogisticRegression(penalty=params[0], dual=True, tol=0.000000001,
                                            C=params[1], fit_intercept=True, intercept_scaling=1,
                                            class_weight=params[2])

    # Gridsearch, not used.
    if params[3]:
        parameters = params[4]
        gs = GridSearchCV(model, parameters, scoring='roc_auc', cv=4)
        gs.fit(x_train, y_train)
        #print "Gridsearch scores:", gs.grid_scores_, "\n"
        print "Best params:", gs.best_params_
        print "Best score:", gs.best_score_
        model = gs.best_estimator_
    else:
        model.fit(x_train, y_train)

    # Predict the sets.
    pred_train = model.predict_proba(train_predict)[:, 1]
    pred_predict = model.predict_proba(x_predict)[:, 1]

    return pred_train, pred_predict


def model_LR(x_train, y_train, train_predict, x_predict, params):
    """
        LR.

        Input:  x_train = training_set,
                y_train = labels for the training set,
                train_predict = the first set to be predicted,
                x_predict = the second set to be predicted,
                params, the parameters for the model: [lr_penalty, lr_C, lr_class_weight].

        Output: [prediction for first set, prediction for second set].
    """

    model = linear_model.LogisticRegression(penalty=params[0], dual=True, tol=0.000000001,
                                            C=params[1], fit_intercept=True, intercept_scaling=1,
                                            class_weight=params[2])

    # Gridsearch, not used.
    if params[3]:
        parameters = params[4]
        gs = GridSearchCV(model, parameters, scoring='roc_auc', cv=4)
        gs.fit(x_train, y_train)
        #print "Gridsearch scores:", gs.grid_scores_, "\n"
        print "Best params:", gs.best_params_
        print "Best score:", gs.best_score_
        model = gs.best_estimator_
    else:
        model.fit(x_train, y_train)

    # Predict the sets.
    pred_train = model.predict_proba(train_predict)[:, 1]
    pred_predict = model.predict_proba(x_predict)[:, 1]

    return pred_train, pred_predict


def model_RIDGE(x_train, y_train, train_predict, x_predict, params):
    """
        Ridge regression.

        Input:  x_train = training_set,
                y_train = labels for the training set,
                train_predict = the first set to be predicted,
                x_predict = the second set to be predicted,
                params, the parameters for the model: [alpha, normalize, max_iter, solver].

        Output: [prediction for first set, prediction for second set].
    """

    model = linear_model.Ridge(alpha=params[0], fit_intercept=True, normalize=params[1],
                               copy_X=True, max_iter=params[2], tol=0.000001, solver=params[3])

    # Gridsearch, not used.
    if params[4]:
        parameters = params[5]
        gs = GridSearchCV(model, parameters, scoring='roc_auc', cv=4)
        gs.fit(x_train, y_train)
        #print "Gridsearch scores:", gs.grid_scores_, "\n"
        print "Best params:", gs.best_params_
        print "Best score:", gs.best_score_
        model = gs.best_estimator_
    else:
        model.fit(x_train, y_train)

    # Predict the sets.
    pred_train = model.predict(train_predict)
    pred_predict = model.predict(x_predict)

    return pred_train, pred_predict


def complete_model_loop(i, train_sets, y_train, predict_sets, sample_size, n_samples, params, models):
    """
        This is the loop needed for the multiprocessing in complete_model.
        It will train multiple models on one sample and return those predictions.
    """

    # Sample the training set if we have a positive sample size.
    if sample_size > 0:
        s_y_train, s_y_cv = cross_validation.train_test_split(y_train,
                                                              test_size=sample_size,
                                                              random_state=SEED+11*i)

        cv_sets = [0 for k in range(len(models))]
        s_train_sets = [0 for k in range(len(models))]

        for j in range(len(models)):
            s_train_sets[j], cv_sets[j] = cross_validation.train_test_split(train_sets[j],
                                                                            test_size=sample_size,
                                                                            random_state=SEED+11*i)
    else:
        s_train_sets, s_y_train = train_sets, y_train

    scores = [[] for k in range(len(models))]
    predict_preds, train_preds = [[] for k in range(len(models))], [[] for k in range(len(models))]

    for k in range(len(models)):
        train_pred, predict_pred = models[k](s_train_sets[k], s_y_train, train_sets[k], predict_sets[k], params[k])

        print "\t Sample %i/%i: \t\t %0.5f" % (i+1, n_samples, roc_auc_score(y_train, train_pred))

        predict_preds[k] = predict_pred
        train_preds[k] = train_pred
        scores[k] = roc_auc_score(y_train, train_pred)

    return train_preds, predict_preds, scores


def complete_model(train_sets, y_train, predict_sets, sample_size, n_samples, params, models,
                   do_ensemble, ensemble_model, ensemble_params):
    """
        This function will train multiple models on multiple training sets (but with the same output vector)
        and return one one-dimensional prediction.

        A "model" consists of a feature or training set, a model function, an appropriate test set and
        a list of parameters for the model function.

        Input:  train_sets = a list of feature sets of a training set,
                y_train = labels for the training set (list),
                predict_sets = a list of corresponding feature sets of a test set whose labels
                               are to be predicted.
                sample_size = sample size in two layer approach (float in ]0,1[),
                n_samples = number of samples in two layer approach,
                params = list of parameter lists for each corresponding model,
                models = list of model functions,
                do_ensemble = whether to use an ensemble model (True) or just a simple average (False),
                ensemble_model = when do_ensemble is True, the first element in this list is the ensemble
                                 model that will be used to combine predictions, otherwise it has no use,
                ensemble_params = corresponding parameters for the ensemble model, when used.

        Output: one prediction for the test set.

        Complete model works as follows:
            1. Take n_samples random samples s_x_train of sample size sample_size.
            2. Train different models on each sample to predict (the base set) x_train and x_predict.
            3. Iterate n_samples times to get an array of predictions for x_train and x_predict.
            4. Average these predictions to get one prediction of x_train and x_predict per model.
            5. Feed the predictions for x_train to another model (or use a simple average).
            6. Use this last model to predict with x_predict.
    """
    # Multiprocessing for 2 procs.
    pool = multiprocessing.Pool(2)
    partial_ = partial(complete_model_loop, train_sets=train_sets, y_train=y_train, predict_sets=predict_sets,
                       sample_size=sample_size, n_samples=n_samples, params=params, models=models)

    res = pool.map_async(partial_, range(n_samples), 1)
    result = np.asarray(res.get())
    pool.close()

    # Gather the results.
    train_preds = np.vstack([np.asarray(result[j, 0, i]) for i in range(len(models)) for j in range(n_samples)]).T
    predict_preds = np.vstack([np.asarray(result[j, 1, i]) for i in range(len(models)) for j in range(n_samples)]).T
    scores = np.hstack([np.asarray(result[j, 2, i]) for i in range(len(models)) for j in range(n_samples)])

    # The combined model: 1. train a new model on the predictions or 2. use a simple average.
    if do_ensemble:
        # Train with the predictions made by the models on the training set (predicted with samples of this set).
        tmp, predict_result = ensemble_model[0](train_preds, y_train, train_preds, predict_preds, ensemble_params[0])
    else:
        # Simple average.
        predict_result = np.mean(predict_preds, axis=1)
        #predict_result = stats.gmean(predict_preds, axis=1)

    return predict_result


def diagnostics(ind_cv, cv_preds, y_cv, full_model, models, train_rest, iter, hist, show_plots):
    """
        This is a diagnostics function which can show some simple interactive plots with labels.
        (Must be enabled in the cross_validate.)
    """

    # Color set for the plots.
    cols = ['r', 'g', 'k', 'b', 'm', 'r', 'b', 'g', 'k', 'm', 'r', 'g', 'k', 'b']

    # We will make a vector of differences (per model): label - predicted.
    # Thus, positive difference <=> label 1 and predicted towards 0.
    #       negative difference <=> label 0 and predicted towards 1.

    # If we have predictions from the full model (thus only one prediction).
    if full_model:
        # The diffs vector.
        cv_diffs = np.asarray([a - b for a, b in zip(y_cv,  cv_preds)])
        df = p.DataFrame(cv_diffs, dtype=object, index=ind_cv)

        # The points of differences.
        points = df.sort(columns=[0]).ix[:, 0].astype('float64')
        # Some labels to easily identify misclassified points (according to trainset).
        labels = ["{%i}:%s\n" % (train_rest.ix[row_nr, 'urlid'],
                                 train_rest.ix[row_nr, 'url']) for row_nr in df.sort(columns=[0]).ix[:, 0].index]

        fig, ax = pl.subplots()

        if hist:
            ax.hist(points, color=cols.pop())
            ax.margins(0.1)
            ax.set_xlim([-1, 1])
            ax.set_ylim([0, 1500])
            ax.set_title("Full model\n AUC: %0.5f" % roc_auc_score(y_cv, cv_preds))
        else:
            ax.plot(points, 'ro', color=cols.pop())
            ax.margins(0.1)
            ax.set_ylim([-1, 1])
            ax.set_title("Full model\n AUC: %0.5f" % roc_auc_score(y_cv, cv_preds))

        if not hist:
            datacursor(axes=ax, point_labels=labels, draggable=True)

        # Save image.
        now = datetime.datetime.now()
        timestamp = now.strftime("%d-%m-%Y")
        pl.savefig('figs/%s_full_model_iter_%i_%s_%0.4f.png' % (timestamp, iter, models, roc_auc_score(y_cv, cv_preds)),
                   bbox_inches=0)

        # Show plot.
        if show_plots:
            pl.show()

    else:
        # One column/plot per model.
        if len(models) > 1:
            ncols = len(models)
        else:
            ncols = 2
        fig, axes = pl.subplots(ncols=ncols)
        for i in range(len(models)):
            # The diffs vector.
            cv_diffs = np.asarray([a - b for a, b in zip(y_cv,  cv_preds[models[i]])])
            df = p.DataFrame(cv_diffs, dtype=object, index=ind_cv)

            # The points of differences.
            points = df.sort(columns=[0]).ix[:, 0].astype('float64')
            # The labels.

            if hist:
                axes[i].hist(points, color=cols.pop())
                axes[i].margins(0.1)
                axes[i].set_xlim([-1, 1])
                axes[i].set_ylim([0, 1500])
                axes[i].set_title("Model %i\n AUC: %0.5f" % (models[i].__name__,
                                                             roc_auc_score(y_cv, cv_preds[models[i]])))
            else:
                labels = ["{%i}:%s\n" % (train_rest.ix[row_nr, 'urlid'],
                                         train_rest.ix[row_nr, 'url'])
                          for row_nr in df.sort(columns=[0]).ix[:, 0].index]
                axes[i].plot(points, 'ro', color=cols.pop())
                axes[i].margins(0.1)
                axes[i].set_ylim([-1, 1])
                axes[i].set_title("Model %i\n AUC: %0.5f" % (models[i].__name__,
                                                             roc_auc_score(y_cv, cv_preds[models[i]])))

            if not hist:
                datacursor(axes=axes[i], point_labels=labels, draggable=True)

        # Save image.
        now = datetime.datetime.now()
        timestamp = now.strftime("%d-%m-%Y")
        pl.savefig('figs/%s_models_iter_%i_%s.png' % (timestamp, iter, models),
                   bbox_inches=0)

        # Show plot(s).
        if show_plots:
            pl.show()


def cross_validate(train_sets, y_train, sample_size, n_samples, cv_size, cv_times, params, full_model,
                   models, do_ensemble, ensemble_models, ensemble_params, train_rest):
    """
        Cross validate the models. Two possibilities:

        1. Cross validate on the full model: ensemble predictions of different models.
        2. Cross validate each model separately.

        Input:  train_sets = a list of feature sets of a training set,
                y_train = list of labels for the training set,
                sample_size = sample size in in two layer approach,
                n_samples = number of samples in two layer approach,
                cv_size = the percentage (as float in ]0,1[) of the training set to be
                          used for validation,
                cv_times = number of times to do cross validation,
                full_model = whether to train on separate models (False) or just the ensemble (True),
                params = list of parameter lists for each corresponding model,
                models = list of model functions,
                do_ensemble = whether to do use an ensemble model (True) or just a simple average (False),
                ensemble_model = when do_ensemble is True, the first element in this list is the ensemble
                                 model that will be used to combine predictions, otherwise it has no use,
                ensemble_params = corresponding parameters for the ensemble model, when used,
                train_rest = some original features to make it easier to use the diagnostics function.

        Output: list of AUCs of the cv_times validations.
    """

    if full_model:
        cv_aucs = []
    else:
        cv_aucs = [[] for i in range(len(models))]

    print "Cross validating ...\n"
    for i in range(cv_times):
        print "\n****************** CV iteration %i/%i. ******************\n" % (i+1, cv_times)

        indices = [k for k in range(len(y_train))]
        s_y_train, s_y_cv, ind_train, ind_cv = cross_validation.train_test_split(y_train, indices,
                                                                                 test_size=cv_size,
                                                                                 random_state=SEED+11*i)

        cv_sets = [0 for k in range(len(models))]
        s_train_sets = [0 for k in range(len(models))]

        for j in range(len(models)):
            s_train_sets[j], cv_sets[j] = cross_validation.train_test_split(train_sets[j],
                                                                            test_size=cv_size,
                                                                            random_state=SEED+11*i)

        # Train the combined full model or each model separately?
        if full_model:
            cv_pred = complete_model(s_train_sets, s_y_train, cv_sets, sample_size, n_samples, params, models,
                                     do_ensemble, ensemble_models, ensemble_params)
            cv_aucs.append(roc_auc_score(s_y_cv, cv_pred))
            print "\n\t Score: %0.5f.\n" % roc_auc_score(s_y_cv, cv_pred)

            # Give predictions cv_pred to diagnostics function.
            #diagnostics(ind_cv, cv_pred, s_y_cv, full_model, models, train_rest, i, hist=True, show_plots=False)
        else:
            # The vector which will contain a prediction per model.
            cv_preds = [[] for j in range(len(models))]

            # Test every model.
            for j in range(len(models)):
                print "Model %s \n\t %s\n" % (models[j].__name__, params[j])

                cv_pred = complete_model([s_train_sets[j]], s_y_train, [cv_sets[j]], sample_size, n_samples,
                                         [params[j]], [models[j]], do_ensemble, ensemble_models, ensemble_params)
                cv_preds[j] = cv_pred
                cv_aucs[j].append(roc_auc_score(s_y_cv, cv_pred))
                print "Score on CV fold: %0.5f.\n" % roc_auc_score(s_y_cv, cv_pred)

            # Give predictions to diagnostics function.
            #diagnostics(ind_cv, cv_preds, s_y_cv, full_model, models, train_rest, i, hist=True, show_plots=False)

    return cv_aucs


def write_submission(predictions, filename):
    print "Writing submission file.\n"
    testfile = p.read_csv('data/test.tsv', sep="\t", na_values=['?'], index_col=1)
    pred_df = p.DataFrame(predictions, index=testfile.index, columns=['label'])
    pred_df.to_csv(filename)


def main():
    # Settings for the TF-IDF processing.
    sets_ids = [0,  # url
                0,  # title
                0,  # body
                0,  # html
                0,  # links
                0]  # htmlbody

    url_sets = [['word', (1, 1), 0.0001, 0.8],         # 1-grams 16774
                ['word', (2, 2), 0.0002, 0.8],         # 2-grams 7500
                ['word', (3, 3), 0.0002, 0.8],         # 3-grams 3700
                ['char', (3, 6), 0.0005, 0.8]]         # orig

    title_sets = [['word', (1, 1), 0.0001, 0.8],       # 1-grams 10245
                  ['word', (2, 2), 0.0002, 0.8],       # 2-grams 7500
                  ['word', (3, 3), 0.0002, 0.8],       # 3-grams 3500
                  ['char', (3, 6), 0.0007, 0.80]]      # orig

    body_sets = [['word', (1, 1), 0.0002, 0.8],        # 1-grams 32000
                 ['word', (2, 2), 0.0010, 0.8],        # 2-grams 45200
                 ['word', (3, 3), 0.0010, 0.8],        # 3-grams 22902
                 ['word', (1, 2), 0.0085, 0.8]]        # orig

    html_sets = [['word', (1, 1), 0.00015, 0.8],       # 1-grams 66074
                 ['word', (2, 2), 0.0010, 0.8],        # 2-grams 116000
                 ['word', (3, 3), 0.0010, 0.8],        # 3-grams 91278
                 ['word', (1, 1), 0.0045, 0.8]]        # orig

    links_sets = [['word', (1, 1), 0.0025, 0.8],       # 1-grams 13789
                  ['word', (2, 2), 0.0020, 0.8],       # 2-grams 35934
                  ['word', (3, 3), 0.0020, 0.8],       # 3-grams 41569
                  ['word', (1, 1), 0.0070, 0.8]]       # orig

    htmlbody_sets = [['word', (1, 1), 0.0002, 0.8],     # 1-grams 72000
                     ['word', (2, 2), 0.0010, 0.8],     # 2-grams 123000
                     ['word', (3, 3), 0.0010, 0.8],     # 3-grams 96500
                     ['word', (1, 1), 0.0065, 0.8]]     # orig

    names = ['url', 'title', 'body', 'html', 'links', 'htmlbody', 'comb']

    ########################################################################################
    #                                   Kernel PCA 1-grams
    sets_ids = [0, 0, 0, 0, 0, 0, 0]
    tfidf_url, tfidf_title = url_sets[sets_ids[0]], title_sets[sets_ids[1]]
    tfidf_body, tfidf_html = body_sets[sets_ids[2]], html_sets[sets_ids[3]]
    tfidf_links, tfidf_htmlbody = links_sets[sets_ids[4]], htmlbody_sets[sets_ids[5]]

    data_1grams = load_data(tfidf_url, tfidf_title, tfidf_body, tfidf_html, tfidf_links, tfidf_htmlbody)

    data_1grams['train_comb'] = csc_matrix(hstack((data_1grams['train_' + 'url'],
                                                   data_1grams['train_' + 'title'],
                                                   data_1grams['train_' + 'body'],
                                                   data_1grams['train_' + 'html'],
                                                   data_1grams['train_' + 'links'],
                                                   data_1grams['train_' + 'htmlbody'])))

    data_1grams['test_comb'] = csc_matrix(hstack((data_1grams['test_' + 'url'],
                                                  data_1grams['test_' + 'title'],
                                                  data_1grams['test_' + 'body'],
                                                  data_1grams['test_' + 'html'],
                                                  data_1grams['test_' + 'links'],
                                                  data_1grams['test_' + 'htmlbody'])))

    train_rest = data_1grams['train_rest']
    test_rest = data_1grams['test_rest']
    y = data_1grams['y']

    train_kpca_1grams = {}
    test_kpca_1grams = {}
    n_components = [800, 500, 100, 600, 1200, 700, 300]  # 800, 500, 100, 600, 1200, 700, 300

    # Get KPCA-transformated data.
    print "\n Doing KPCA transformations for 1-grams with %s components ...\n" % n_components
    for j in range(len(names)):
        # Try to get from dump first.
        try:
            train_kpca_1grams[names[j]] = pickle.load(open("dump/kpca_train_%s_%i_comp_%i.pkl" % (names[j],
                                                                                                  sets_ids[j],
                                                                                                  n_components[j]),
                                                           "rb"))
            test_kpca_1grams[names[j]] = pickle.load(open("dump/kpca_test_%s_%i_comp_%i.pkl" % (names[j],
                                                                                                sets_ids[j],
                                                                                                n_components[j]),
                                                          "rb"))
        # If it fails, then fit again.
        except:
            print "\t Could not read KPCA for %s from dump, fitting ..." % names[j]
            kpca = KernelPCA(n_components=n_components[j], kernel='linear')

            kpca.fit(vstack((data_1grams['train_' + names[j]], data_1grams['test_' + names[j]])))

            train_kpca_1grams[names[j]] = kpca.transform(data_1grams['train_' + names[j]])

            test_kpca_1grams[names[j]] = kpca.transform(data_1grams['test_' + names[j]])

            with open("dump/kpca_train_%s_%i_comp_%i.pkl" % (names[j], sets_ids[j], n_components[j]), "wb") as f:
                pickle.dump(train_kpca_1grams[names[j]], f, pickle.HIGHEST_PROTOCOL)

            with open("dump/kpca_test_%s_%i_comp_%i.pkl" % (names[j], sets_ids[j], n_components[j]), "wb") as f:
                pickle.dump(test_kpca_1grams[names[j]], f, pickle.HIGHEST_PROTOCOL)

            print "\t Dumped."

    print " Done.\n"

    ########################################################################################
    #                                   Kernel PCA 2-grams
    sets_ids = [1, 1, 1, 1, 1, 1, 1]
    tfidf_url, tfidf_title = url_sets[sets_ids[0]], title_sets[sets_ids[1]]
    tfidf_body, tfidf_html = body_sets[sets_ids[2]], html_sets[sets_ids[3]]
    tfidf_links, tfidf_htmlbody = links_sets[sets_ids[4]], htmlbody_sets[sets_ids[5]]

    data_2grams = load_data(tfidf_url, tfidf_title, tfidf_body, tfidf_html, tfidf_links, tfidf_htmlbody)

    data_2grams['train_comb'] = csc_matrix(hstack((data_2grams['train_' + 'url'],
                                                   data_2grams['train_' + 'title'],
                                                   data_2grams['train_' + 'body'],
                                                   data_2grams['train_' + 'html'],
                                                   data_2grams['train_' + 'links'],
                                                   data_2grams['train_' + 'htmlbody'])))

    data_2grams['test_comb'] = csc_matrix(hstack((data_2grams['test_' + 'url'],
                                                  data_2grams['test_' + 'title'],
                                                  data_2grams['test_' + 'body'],
                                                  data_2grams['test_' + 'html'],
                                                  data_2grams['test_' + 'links'],
                                                  data_2grams['test_' + 'htmlbody'])))

    # Get KPCA-transformed data.
    train_kpca_2grams = {}
    test_kpca_2grams = {}
    n_components = [1300, 1100, 500, 500, 500, 500, 300] # 1300, 1100, 500, 500, 500, 500, 300
    print "\n Doing KPCA transformations for 2-grams with %s components ...\n" % n_components
    for j in range(len(names)):
        # Try to get from dump first.
        try:
            train_kpca_2grams[names[j]] = pickle.load(open("dump/kpca_train_%s_%i_comp_%i.pkl" % (names[j],
                                                                                                  sets_ids[j],
                                                                                                  n_components[j]),
                                                           "rb"))
            test_kpca_2grams[names[j]] = pickle.load(open("dump/kpca_test_%s_%i_comp_%i.pkl" % (names[j],
                                                                                                sets_ids[j],
                                                                                                n_components[j]),
                                                          "rb"))
        # If it fails, then fit again.
        except:
            print "\t Could not read KPCA for %s from dump, fitting ..." % names[j]
            kpca = KernelPCA(n_components=n_components[j], kernel='linear')

            kpca.fit(vstack((data_2grams['train_' + names[j]], data_2grams['test_' + names[j]])))

            train_kpca_2grams[names[j]] = kpca.transform(data_2grams['train_' + names[j]])

            test_kpca_2grams[names[j]] = kpca.transform(data_2grams['test_' + names[j]])

            with open("dump/kpca_train_%s_%i_comp_%i.pkl" % (names[j], sets_ids[j], n_components[j]), "wb") as f:
                pickle.dump(train_kpca_2grams[names[j]], f, pickle.HIGHEST_PROTOCOL)

            with open("dump/kpca_test_%s_%i_comp_%i.pkl" % (names[j], sets_ids[j], n_components[j]), "wb") as f:
                pickle.dump(test_kpca_2grams[names[j]], f, pickle.HIGHEST_PROTOCOL)

            print "\t Dumped."

    print " Done.\n"

    ########################################################################################
    #                                   Original set.
    sets_ids = [3, 3, 3, 3, 3, 3]
    tfidf_url, tfidf_title = url_sets[sets_ids[0]], title_sets[sets_ids[1]]
    tfidf_body, tfidf_html = body_sets[sets_ids[2]], html_sets[sets_ids[3]]
    tfidf_links, tfidf_htmlbody = links_sets[sets_ids[4]], htmlbody_sets[sets_ids[5]]

    data_orig = load_data(tfidf_url, tfidf_title, tfidf_body, tfidf_html, tfidf_links, tfidf_htmlbody)

    models = []
    params = []
    train_sets = []
    test_sets = []

    """
        Here the models are constructed by:
            1. Appending the feature set to train_sets (and a corresponding test set to test_sets if applicable).
            2. Appending a model function to models.
            3. Appending parameters for the model in params (format as in the docs of the model function).
    """
    # Model 1: LR with Chi^2 selection on original feature set (without dimensionality reduction).
    comb_orig = csc_matrix(hstack((data_orig['train_' + 'url'],
                                   data_orig['train_' + 'title'],
                                   data_orig['train_' + 'body'],
                                   data_orig['train_' + 'links'],
                                   data_orig['train_' + 'htmlbody'])))

    comb_orig_test = csc_matrix(hstack((data_orig['test_' + 'url'],
                                        data_orig['test_' + 'title'],
                                        data_orig['test_' + 'body'],
                                        data_orig['test_' + 'links'],
                                        data_orig['test_' + 'htmlbody'])))

    train_sets.append(comb_orig)
    test_sets.append(comb_orig_test)
    models.append(model_Chi2LR)
    params.append(['l2', 0.32, None, False, 0, 80,
                   'Original model.'])

    # Model 2: LR on PCA transform of 1-grams.
    comb_1grams = csc_matrix(np.hstack((train_kpca_1grams['url'],
                                        train_kpca_1grams['title'],
                                        train_kpca_1grams['body'],
                                        train_kpca_1grams['html'],
                                        train_kpca_1grams['links'],
                                        train_kpca_1grams['htmlbody'])))

    comb_1grams_test = csc_matrix(np.hstack((test_kpca_1grams['url'],
                                             test_kpca_1grams['title'],
                                             test_kpca_1grams['body'],
                                             test_kpca_1grams['html'],
                                             test_kpca_1grams['links'],
                                             test_kpca_1grams['htmlbody'])))

    train_sets.append(comb_1grams)
    test_sets.append(comb_1grams_test)
    models.append(model_LR)
    params.append(['l2', 0.30, None, False, 0,
                   'KPCA of 1-grams.'])

    ## Model 3: LR on PCA transform of all 1-grams + the 2-gram of body.
    comb_1grams = csc_matrix(np.hstack((train_kpca_1grams['url'],
                                        train_kpca_1grams['title'],
                                        train_kpca_1grams['body'],
                                        train_kpca_1grams['html'],
                                        train_kpca_1grams['links'],
                                        train_kpca_1grams['htmlbody'],
                                        train_kpca_2grams['body'])))

    comb_1grams_test = csc_matrix(np.hstack((test_kpca_1grams['url'],
                                             test_kpca_1grams['title'],
                                             test_kpca_1grams['body'],
                                             test_kpca_1grams['html'],
                                             test_kpca_1grams['links'],
                                             test_kpca_1grams['htmlbody'],
                                             test_kpca_2grams['body'])))

    train_sets.append(comb_1grams)
    test_sets.append(comb_1grams_test)
    models.append(model_LR)
    params.append(['l2', 0.31, None, False, 0,
                   'KPCA 1-grams + body 2-grams.'])

    ########################################################################################
    #                                Cross validation per model.
    # cv_settings = (sample_size, sample_times, cv_size, cv_times)
    cv_settings = (0.20, 6, 0.20, 10)
    ensemble_models = []
    ensemble_params = []
    do_ensemble = False

    print "\n---------------------------------------"
    print "Cross validating per model." % models
    print("Settings: [%i times %0.2f] CV with per model [%i times %0.2f]" %
          (cv_settings[3], cv_settings[2], cv_settings[1], cv_settings[0]))

    aucs = cross_validate(train_sets=train_sets, y_train=y, sample_size=cv_settings[0], n_samples=cv_settings[1],
                          cv_size=cv_settings[2], cv_times=cv_settings[3], params=params, full_model=False,
                          models=models, do_ensemble=do_ensemble, ensemble_models=ensemble_models,
                          ensemble_params=ensemble_params, train_rest=train_rest)

    print "\n---------------------------------------"

    print "\nResults [SEED=%s]" % SEED
    print("\t Settings: [%i times %0.2f] CV with per model [%i times %0.2f]\n" %
          (cv_settings[3], cv_settings[2], cv_settings[1], cv_settings[0]))
    for i in range(len(models)):
        auc = np.asarray(aucs[i])
        print "\t Model %s with params %s and dataset size %i:" % (models[i].__name__, params[i], train_sets[i].shape[1])
        print "\t\t %0.5f (+/- %0.5f) = [%0.5f, %0.5f]" % (np.mean(auc), np.std(auc),
                                                           np.mean(auc) - np.std(auc),
                                                           np.mean(auc) + np.std(auc))

        print "\t\t [Min, Median, Max] = [%0.5f, %0.5f, %0.5f]\n" % (min(auc), np.median(auc), max(auc))

    #######################################################################################
    #                                Cross validation on full model.
    # cv_settings = (sample_size, sample_times, cv_size, cv_times)
    cv_settings = (0.20, 6, 0.20, 10)
    ensemble_models = [model_RIDGE]
    ensemble_params = [[2, True, None, 'auto', False, 0]]

    ensemble_models = [ensemble_models[0]]
    ensemble_params = [ensemble_params[0]]
    do_ensemble = False

    print "\n---------------------------------------"
    print "Cross validating full model with models. [SEED=%s]" % SEED
    print("Settings: [%i times %0.2f] CV with per model [%i times %0.2f]" %
          (cv_settings[3], cv_settings[2], cv_settings[1], cv_settings[0]))

    aucs = cross_validate(train_sets=train_sets, y_train=y, sample_size=cv_settings[0], n_samples=cv_settings[1],
                          cv_size=cv_settings[2], cv_times=cv_settings[3], params=params, full_model=True,
                          models=models, do_ensemble=do_ensemble, ensemble_models=ensemble_models,
                          ensemble_params=ensemble_params, train_rest=train_rest)

    print "\n---------------------------------------"

    print "\nResults [SEED=%s]" % SEED
    print("\t Settings: [%i times %0.2f] CV with per model [%i times %0.2f]\n" %
          (cv_settings[3], cv_settings[2], cv_settings[1], cv_settings[0]))
    for i in range(len(models)):
        print "\t Model %s with params %s and dataset size %i:" % (models[i].__name__,
                                                                   params[i],
                                                                   train_sets[i].shape[1])

    print "\n\t [%s] Ensemble model %s with params %s.\n" % (do_ensemble, ensemble_models[0].__name__,
                                                             ensemble_params[0])

    print "\n\t\t RESULT: %0.5f (+/- %0.5f) = [%0.5f, %0.5f]" % (np.mean(aucs), np.std(aucs),
                                                                 np.mean(aucs) - np.std(aucs),
                                                                 np.mean(aucs) + np.std(aucs))

    print "\t\t [Min, Median, Max] = [%0.5f, %0.5f, %0.5f]\n" % (min(aucs), np.median(aucs), max(aucs))

    print "\n\n Raw AUCs: \n %s." % aucs

    #######################################################################################
    #                       Training on all data, predicting the test set.
    # Sample size, no. of samples.
    #test_params = (0.30, 6)
    #
    ### Test set prediction.
    #print "\n---------------------------------------"
    #print "Predicting the test set with full model and submodels %s." % [model.__name__ for model in models]
    #
    #print "\n---------------------------------------"
    #print("Settings [SEED=%s]\n" % SEED)
    #
    #for i in range(len(models)):
    #    print "\t Model %s with params %s and dataset size %i:" % (models[i].__name__,
    #                                                               params[i],
    #                                                               train_sets[i].shape[1])
    #
    #print "\n\t [%s] Ensemble model %s with params %s.\n" % (do_ensemble,
    #                                                         ensemble_models[0].__name__,
    #                                                         ensemble_params[0])
    #
    ## Get predictions on test set for combined model.
    #testpred = complete_model(train_sets, y, test_sets, test_params[0],
    #                          test_params[1], params, models, do_ensemble,
    #                          ensemble_models, ensemble_params)
    #
    ## Get timestamp for submission filename.
    #now = datetime.datetime.now()
    #timestamp = now.strftime("%d-%m-%Y-%H:%M")
    #filename = 'submissions/submission_%0.2f_%i_%i_%s_%s.csv' % (test_params[0], test_params[1], len(models), timestamp, SEED)
    ## Write submission file.
    #write_submission(testpred, filename)
    #
    #print "\n Written submission file %s." % filename

    print "\n---------------------------------------\n\n"

if __name__ == '__main__':
    main()
