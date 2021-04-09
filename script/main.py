import nltk
import numpy as np
import util
import validation
import preprocess
import sys
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GroupShuffleSplit
from skmultilearn.problem_transform import BinaryRelevance
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from skmultilearn.adapt import BRkNNaClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
np.set_printoptions(threshold=sys.maxsize)

def try_knn(train, test):
    x_train, y_train = train[['sentence','heading']], train[['abstract']]
    x_test, y_test = test[['sentence','heading']], test[['abstract']]

    for column in ["sentence","heading"]:
      x_train[column] = x_train[column].apply(preprocess.clean)
      x_test[column] = x_test[column].apply(preprocess.clean)

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
    y_pred = pipe.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('KNN with TFIDF accuracy = ' + str(acc * 100) + '%')

    print(validation.validate(x_test,y_test,y_pred))
    return y_pred

def classify(df_train, df_test):
    x_train, y_train = df_train[['sentence','heading']], df_train[['labels']]
    x_test, y_test = df_test[['sentence','heading']], df_test[['labels']]

    for column in ["sentence","heading"]:
      x_train[column] = x_train[column].apply(preprocess.clean)
      x_test[column] = x_test[column].apply(preprocess.clean)

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
    y_pred = pipe.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('fucken accuracy = ' + str(acc * 100) + '%')
    # print(validation.validate_more(x_test,y_test,y_pred))

    prediction = x_test
    prediction['labels'] = y_test
    prediction['pred'] = y_pred
    # prediction.to_csv('check.csv')

def dosumn():
    #part 1: sentence removal, pick only summary-worthy sentence
    # data = util.get_parser_data("train_set.csv")
    # tr, te = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7).split(data, groups=data['filename']))
    # train = data.iloc[tr]
    # test = data.iloc[te]

    train = util.get_parser_data("train_set.csv")
    test = util.get_parser_data("test_file.csv")

    train['sentence'] = train['sentence'].apply(preprocess.clean)
    test['summary_worth'] = try_knn(train, test)
    print(" ".join(test.loc[test['summary_worth']== 1,'sentence']))

    #part 2: labelling
    # c_train = train[train['abstract'] == 1].copy()
    # c_test = test[test['summary_worth'] == 1].copy()
    
    # classify(c_train, c_test)
    
    # df_result.to_csv('check.csv')

if __name__ == "__main__":  
    dosumn()