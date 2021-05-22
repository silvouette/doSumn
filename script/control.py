import util
import rank
import os
import pandas as pd
from sklearn import metrics, svm
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

path = os.path.dirname(os.getcwd())
rm_path = os.path.join(path, "json_rm/")
cl_path = os.path.join(path, "json_cl/")

train = util.get_parser_data("train_set.csv")
for column in ["sentence","heading"]: #sentence cleaning includes lowercasing, removing extra parts and numbers, punctuations. Refer to util.py)
  train[column+"_x"] = train[column].apply(util.clean)

x_train, y_train = train[['sentence_x','heading_x']], train[['abstract']]
transformer = FeatureUnion([ #TF-IDF 
              ('sentence_tfidf', 
                Pipeline([('extract_field',
                            FunctionTransformer(lambda x: x['sentence_x'], validate=False)),
                          ('tfidf', TfidfVectorizer(ngram_range=[1,2]))])),
              ('heading_tfidf', 
                Pipeline([('extract_field', 
                            FunctionTransformer(lambda x: x['heading_x'], validate=False)),
                          ('tfidf', TfidfVectorizer())]))]) 

pipe = Pipeline(steps=[ #Pipeline includes TF-IDF and classifier. Binary SVM classifier used.
                  ('tfidf', transformer),
                  ('classifier', svm.SVC(class_weight='balanced', kernel='rbf', C=100))
                  ])
pipe.fit(x_train, y_train.values.ravel())

#2nd class
xc_train, yc_train = train[['sentence','sentence_x','heading_x']], train[['labels']]
transformerc = FeatureUnion([ #TF-IDF
                ('sentence_tfidf', 
                  Pipeline([('extract_field',
                              FunctionTransformer(lambda x: x['sentence_x'], validate=False)),
                            ('tfidf', TfidfVectorizer(ngram_range=[1,2]))])),
                ('heading_tfidf', 
                  Pipeline([('extract_field', 
                              FunctionTransformer(lambda x: x['heading_x'], validate=False)),
                            ('tfidf', TfidfVectorizer())]))]) 
pipec = Pipeline(steps=[ #Pipeline includes TF-IDF and classifier. Multiclass one-vs-rest SVM classifier used. Other parameters shown.
                    ('tfidf', transformerc),
                    ('classifier', svm.SVC(class_weight='balanced', kernel='rbf', decision_function_shape='ovr',  gamma=0.005, C=1000))
                    ])
pipec.fit(xc_train, yc_train.values.ravel()) #data fitting


def removes(test): #classifying sentences to class 1 if summary-worthy, else 0
    x_test = test[['sentence_x','heading_x']]
    y_pred = pipe.predict(x_test) #predict labels
    # accuracy = metrics.accuracy_score(test['abstract'].values, y_pred)*100
    # print(accuracy,",")
    return y_pred

def classify(df_test): #classifying sentences into rhetorical roles [background, topic, method, dataset, result, conclusion, suggestion]
    x_test, y_test = df_test[['sentence','sentence_x','heading_x']], df_test[['labels']] #assigning test set
    y_pred = pipec.predict(x_test)
    prediction = x_test
    prediction['labels'], prediction['pred'] = y_test, y_pred #new dataframe to include test expected value and predicted value

    accuracy = metrics.accuracy_score(y_test, y_pred)*100
    print(accuracy,",")
    return prediction

def dosumn(filename):
    # data prep & clean
    test = util.get_parser_data("test/"+filename)

    for column in ["sentence","heading"]: #sentence cleaning includes lowercasing, removing extra parts and numbers, punctuations. Refer to util.py)
      test[column+"_x"] = test[column].apply(util.clean)

    # part 1: sentence removal, pick only summary-worthy sentence
    test['summary_worth']= removes(test)
    name = os.path.splitext(filename)[0]
    test.to_json(rm_path+name+".json")

    #part 2: labelling
    c_test = test[test['summary_worth'] == 1].copy()
    classed = classify(c_test) #classed is dataset to display in flask as result of 2nd classifiction. cl_acc is 2nd classification accuracy
    classed.to_json(cl_path+name+".json")

if __name__ == "__main__":  
  for filename in os.listdir('../test'):
    dosumn(filename)
