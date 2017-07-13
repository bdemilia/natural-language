import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "..."

df = pd.read_csv(path + "\\train.csv").fillna("")
df.head()

df.info()
df.shape
df.groupby("is_duplicate")['id'].count().plot.bar()

dfs = df[0:2500]
dfs.groupby("is_duplicate")['id'].count().plot.bar()

dfq1, dfq2 = dfs[['qid1', 'question1']], dfs[['qid2', 'question2']]
dfq1.columns = ['qid1', 'question']
dfq2.columns = ['qid2', 'question']

dfqa = pd.concat((dfq1, dfq2), axis=0).fillna("")
nrows_for_q1 = dfqa.shape[0]/2
dfqa.shape

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
mq1 = TfidfVectorizer(max_features = 256).fit_transform(dfqa['question'].values)
mq1
