# from nltk.corpus import stopwords
# import string
#
# # nltk.download()
# print(stopwords.words('english')[0:10])
# test_sentence = 'This is my first test string. Wow!! we are doing just fine.'
# no_punctuation = [char for char in test_sentence if char not in string.punctuation]
# print(no_punctuation)
# no_punctuation = ''.join(no_punctuation)
# print(no_punctuation)
# no_punctuation.split()
# clean_sentence = [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]
# print(clean_sentence)

# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer()
# doc1 = 'Hi How are you'
# doc2 = 'Today is a very very very pleasant day and we can have some fun fun fun'
# doc3 = 'This was an amazing experience'
# listDoc = [doc1, doc2, doc3]
# bag_of_words = vectorizer.fit(listDoc)
# bag_of_words = vectorizer.transform(listDoc)
# print(bag_of_words)
# print(vectorizer.vocabulary_.get('very'))
# print(vectorizer.vocabulary_.get('fun'))



import pandas as pd
import string
from pprint import pprint
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline



df = pd.DataFrame({'response': ['ham', 'ham', 'spam', 'ham', 'ham'], 'message':['go untill...', 'ok lar...',
                'Free entry in 2 a wkly comp', 'U dun say so early', 'nahh i dont think...']})
# print(df)
pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier)])
parameters = {'tfidf_use_idf': (True, False)}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
pprint(parameters)
t0 = time()
grid_search.fit(df['message'], df['response'])
print('done in', time() - t0)
print()
