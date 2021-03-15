import pandas as pd
import nltk
import string
import re
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
pd.options.mode.chained_assignment = None

def get_parser_data(filename):
    data = pd.read_csv('../' + filename)
    return data

def preprocessing(text):
    noise = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'yang', 'akan', 'dan']
    text = text.translate(str.maketrans("","",string.punctuation)).strip().lower()
    text = re.sub(r"\d+", "", text)
    text = ' '.join(w for w in text.split() if w not in noise)
    return text

def try_knn():
    df = get_parser_data("dataset.csv")
    x = df[['sentence','heading']]
    y = df[['abstract']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    for column in ["sentence","heading"]:
        x_train[column] = x_train[column].apply(preprocessing)
        x_test[column] = x_test[column].apply(preprocessing)
    
    transformer = FeatureUnion([
                ('sentence_tfidf', 
                  Pipeline([('extract_field',
                              FunctionTransformer(lambda x: x['sentence'], validate=False)),
                            ('tfidf', TfidfVectorizer())])),
                ('heading_tfidf', 
                  Pipeline([('extract_field', 
                              FunctionTransformer(lambda x: x['heading'], validate=False)),
                            ('tfidf', TfidfVectorizer())]))]) 

    pipe = Pipeline(steps=[
                    ('tfidf', transformer),
                    ('classifier', KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', leaf_size=30, p=2,
                                                        metric='cosine', metric_params=None, n_jobs=1))
                    ])
    pipe.fit(x_train, y_train.values.ravel())
    y_test_pred = pipe.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_test_pred)
    print('KNN with TFIDF accuracy = ' + str(acc * 100) + '%')

if __name__ == "__main__":  
    try_knn()