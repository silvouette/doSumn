import nltk
import numpy as np
import util
import validation
import preprocess
import sys
import pandas as pd
from sklearn import metrics, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score, cross_val_predict
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
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
                            ('tfidf', TfidfVectorizer(ngram_range=[1,2]))])),
                ('heading_tfidf', 
                  Pipeline([('extract_field', 
                              FunctionTransformer(lambda x: x['heading'], validate=False)),
                            ('tfidf', TfidfVectorizer())]))]) 

    pipe = Pipeline(steps=[
                    ('tfidf', transformer),
                    ('classifier', svm.SVC(class_weight='balanced', kernel='linear', C=10))
                    ])
    pipe.fit(x_train, y_train.values.ravel())
    y_pred = pipe.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('SVM accuracy = ' + str(acc * 100) + '%')

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
                            ('tfidf', TfidfVectorizer(ngram_range=[1,3], max_features=1000))])),
                ('heading_tfidf', 
                  Pipeline([('extract_field', 
                              FunctionTransformer(lambda x: x['heading'], validate=False)),
                            ('tfidf', TfidfVectorizer())]))]) 
    pipe = Pipeline(steps=[
                    ('tfidf', transformer),
                    ('classifier', svm.SVC(class_weight='balanced', kernel='poly', decision_function_shape='ovr',  gamma=10, C=0.1))
                    ])
    pipe.fit(x_train, y_train.values.ravel())
    y_pred = pipe.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('fucken accuracy = ' + str(acc * 100) + '%')

    prediction = x_test
    prediction['labels'] = y_test
    prediction['pred'] = y_pred

    print(prediction)

    # validation.validate_more(prediction)
    # print(y_pred)
    return prediction
    # prediction.to_csv('check.csv')

def dosumn():
    #part 1: sentence removal, pick only summary-worthy sentence
    train = util.get_parser_data("train_set.csv")
    test = util.get_parser_data("4.csv")

    train['sentence'] = train['sentence'].apply(preprocess.clean)
    test['summary_worth'] = try_knn(train, test)
    # print(" ".join(test.loc[test['summary_worth']== 1,'sentence']))

    # #part 2: labelling
    c_train = train[train['abstract'] == 1].copy()
    c_test = test[test['summary_worth'] == 1].copy()
    
    classify(c_train, c_test)
    
    # df_result.to_csv('check.csv')

if __name__ == "__main__":  
    dosumn()