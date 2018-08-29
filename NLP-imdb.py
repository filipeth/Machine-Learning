import pandas as pd
import numpy as np
import os.path
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


csvfile = os.path.join('/Volumes', 'Filipe Theodoro', 'Artificial Inteligence',
                       'Projects', 'assigment 8', 'imdb_labelled.txt')
df = pd.read_table(csvfile, header=None, names=['messages', 'label'])
# print(df.head())
df['length'] = df['messages'].apply(len)
print(df.shape)
vectorizer = CountVectorizer()
# print(df.info())

# print(df.groupby('label').describe())
# print(df[df['length'] > 50]['messages'].iloc[0])


def eliminate_stopwords(df1):
    word = []
    for frase in df1['messages']:
        # print(frase)
        clean = [word for word in frase.split() if word.lower() not in stopwords.words('english')]
        # print(' '.join(clean))
        word.append(' '.join(clean))
    return word


teste = eliminate_stopwords(df)
bag_of_words = vectorizer.fit(teste)
bag_of_words = vectorizer.transform(teste)
# print(bag_of_words.shape)
tfidf_transformer = TfidfTransformer().fit(bag_of_words)
comment_tfidf = tfidf_transformer.transform(bag_of_words)
# print(comment_tfidf.shape)
spam_detect = MultinomialNB().fit(comment_tfidf, df['label'])
message = df['messages'][11]
bag_of_words_for_message = vectorizer.transform([message])
tfid = tfidf_transformer.transform(bag_of_words_for_message)
predicted = spam_detect.predict(tfid)
print('predicted', predicted[0])
print('expected ', df['label'][11])
# print(metrics.accuracy_score(df['messages'], predicted))
print(predicted[0:3])

