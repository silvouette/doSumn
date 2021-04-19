import os
import nltk
import numpy as np
import util
import validation
import preprocess
import sys
import pandas as pd
from sklearn import metrics, svm
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score, cross_val_predict
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
np.set_printoptions(threshold=sys.maxsize)

def try_knn(train, test):
    x_train, y_train = train[['sentence_x','heading_x']], train[['abstract']]
    x_test, y_test = test[['sentence_x','heading_x']], test[['abstract']]

    transformer = FeatureUnion([
                ('sentence_tfidf', 
                  Pipeline([('extract_field',
                              FunctionTransformer(lambda x: x['sentence_x'], validate=False)),
                            ('tfidf', TfidfVectorizer(ngram_range=[1,2]))])),
                ('heading_tfidf', 
                  Pipeline([('extract_field', 
                              FunctionTransformer(lambda x: x['heading_x'], validate=False)),
                            ('tfidf', TfidfVectorizer())]))]) 

    pipe = Pipeline(steps=[
                    ('tfidf', transformer),
                    ('classifier', svm.SVC(class_weight='balanced', kernel='linear', C=10))
                    ])
    pipe.fit(x_train, y_train.values.ravel())
    y_pred = pipe.predict(x_test)
    # acc = metrics.accuracy_score(y_test, y_pred)
    # print(str(acc * 100) + '%')
    # print(validation.validate(x_test,y_test,y_pred))
    return y_pred

def classify(df_train, df_test):
    x_train, y_train = df_train[['sentence','sentence_x','heading_x']], df_train[['labels']]
    x_test, y_test = df_test[['sentence','sentence_x','heading_x']], df_test[['labels']]

    transformer = FeatureUnion([
                ('sentence_tfidf', 
                  Pipeline([('extract_field',
                              FunctionTransformer(lambda x: x['sentence_x'], validate=False)),
                            ('tfidf', TfidfVectorizer(ngram_range=[1,3], max_features=2000))])),
                ('heading_tfidf', 
                  Pipeline([('extract_field', 
                              FunctionTransformer(lambda x: x['heading_x'], validate=False)),
                            ('tfidf', TfidfVectorizer())]))]) 
    pipe = Pipeline(steps=[
                    ('tfidf', transformer),
                    ('classifier', svm.SVC(class_weight='balanced', kernel='poly', decision_function_shape='ovr',  gamma=10, C=0.1))
                    ])
    pipe.fit(x_train, y_train.values.ravel())
    y_pred = pipe.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    accuracy = acc * 100
    prediction = x_test
    prediction['labels'] = y_test
    prediction['pred'] = y_pred

    result = validation.validate_more(prediction)
    return result

def dosumn(filename):
    #part 1: sentence removal, pick only summary-worthy sentence
    train = util.get_parser_data("train_set.csv")
    test = util.get_parser_data("test/"+filename)

    for column in ["sentence","heading"]:
      train[column+"_x"] = train[column].apply(preprocess.clean)
      test[column+"_x"] = test[column].apply(preprocess.clean)

    test['summary_worth'] = try_knn(train, test)
    # text = " ".join(test.loc[test['summary_worth']== 1,'sentence'])
    #part 2: labelling
    c_train = train[train['abstract'] == 1].copy()
    c_test = test[test['summary_worth'] == 1].copy()
    
    res = classify(c_train, c_test)

    return res

# if __name__ == "__main__":  
#     dosumn("class - 1.csv")