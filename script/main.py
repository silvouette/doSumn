import nltk
import numpy as np
import util
import validation
import preprocess
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from skmultilearn.problem_transform import BinaryRelevance
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def try_knn(df_train, df_test):
    x_train, y_train = df_train[['sentence','heading']], df_train[['abstract']]
    x_test, y_test = df_test[['sentence','heading']], df_test[['abstract']]

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

    # print(validation.validate(x_test,y_test,y_pred))
    return y_pred

def classify(df_train, df_test):
    x_train, y_train = df_train[['sentence','heading']], df_train[['bg','tp','mt','dt','rs','cl','sg']]
    x_test, y_test = df_test[['sentence','heading']], df_test[['bg','tp','mt','dt','rs','cl','sg']]

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
                    ('classifier', BinaryRelevance(KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', leaf_size=30, p=2,
                                                        metric='cosine', metric_params=None, n_jobs=1)))
                    ])
    pipe.fit(x_train, y_train.values.ravel())
    y_pred = pipe.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('fucken accuracy = ' + str(acc * 100) + '%')

def dosumn():
    #part 1: sentence removal, pick only summary-worthy sentence
    df_train = util.get_parser_data("train_set.csv")
    df_test = util.get_parser_data("test_file.csv")

    df_test['summary_worth'] = try_knn(df_train, df_test)

    #part 2: labelling
    c_df_train = df_train[df_train['abstract'] == 1].copy()
    c_df_test = df_test[df_test['summary_worth'] == 1].copy()
    
    # classify(c_df_train, c_df_test)

if __name__ == "__main__":  
    dosumn()