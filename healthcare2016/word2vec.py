# -*- coding: utf-8 -*-
"""

The same code as word2vec.py, but different input data and only two models instead four.
Competition: Healthcare Data Analytics Challenge
Author: Bikash
Team:  UiS

"""


from config import *

import gensim
import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from nltk.stem.snowball import SnowballStemmer, PorterStemmer
import nltk
from time import time
import re
import os
import math as m
import pandas as pd
from gensim import models


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
df_all=pd.read_csv(PROCESSINGTEXT_DIR+"/train.csv", encoding="ISO-8859-1")


def replace_nan(s):
        if pd.isnull(s)==True:
                s=""
        return s

p = df_all.keys()
for i in range(len(p)):
    print (p[i])






df_all['Title'] = df_all['Title'].map(lambda x:replace_nan(x))
df_all['Question'] = df_all['Question'].map(lambda x:replace_nan(x))

st = df_all['Question']
pt = df_all['Title']

print ("first vocab")
#st conc pt conc pd vocab
t1 = list()
for i in range(len(st)):
    p = st[i].split()+pt[i].split()
    t1.append(p)




#model0 = gensim.models.Word2Vec(t, sg=1, window=10, sample=1e-5, negative=5, size=300)
model1 = gensim.models.Word2Vec(t1, sg=1, window=10, sample=1e-5, negative=5, size=300)
print ("model prepared")
#model2 = gensim.models.Word2Vec(t2, sg=1, window=10, sample=1e-5, negative=5, size=300)
#model3 = gensim.models.Word2Vec(t3, sg=1, window=10, sample=1e-5, negative=5, size=300)
print ("model prepared")
#model4 = gensim.models.Word2Vec(t, sg=0,  hs=1, window=10,   size=300)
#model5 = gensim.models.Word2Vec(t1, sg=0, hs=1,window=10,   size=300)
#model6 = gensim.models.Word2Vec(t2, sg=0, hs=1, window=10,   size=300)
#model7 = gensim.models.Word2Vec(t3, sg=0, hs=1,window=10,   size=300)



#model_list=[model0,model1,model2,model3]   #,model4  ,model5,model6,model7]
model_list=[model1]
n_sim=list()

for model in model_list:
    print ("model features calculation")
    n_sim_pt=list()
    for i in range(len(st)):
        w1=st[i].split()
        w2=pt[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_pt.append(0)
        else:    
            n_sim_pt.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_pt)
    
    n_sim_pd=list()
    for i in range(len(st)):
        w1=st[i].split()
        w2=pt[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_pd.append(0)
        else:    
            n_sim_pd.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_pd)

    n_sim_at=list()
    for i in range(len(st)):
        w1=st[i].split()
        w2=pt[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_at.append(0)
        else:    
            n_sim_at.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_at)

    n_sim_all=list()
    for i in range(len(st)):
        w1=st[i].split()
        w2=pt[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_all.append(0)
        else:    
            n_sim_all.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_all)


    
    n_sim_ptpd=list()
    for i in range(len(st)):
        w1=pt[i].split()
        w2=st[i].split()
        d1=[]
        d2=[]
        for j in range(len(w1)):
            if w1[j] in model.vocab:
                d1.append(w1[j])
        for j in range(len(w2)):
            if w2[j] in model.vocab:
                d2.append(w2[j])
        if d1==[] or d2==[]:
            n_sim_ptpd.append(0)
        else:    
            n_sim_ptpd.append(model.n_similarity(d1,d2))
    n_sim.append(n_sim_ptpd)
    print ("model features done")

st_names="3"    
#for j in range(len(n_sim)):   
name_list=list([2,3])
for j in range(len(n_sim)):   
    #st_names=j
    df_all["word2vec_"+str(name_list[j])]=n_sim[j]
    st_names.append("word2vec_"+str(name_list[j]))
    


b=df_all[st_names]
b.to_csv(FEATURES_DIR+"/df_word2vec_wo_google_dict.csv", index=False) 




