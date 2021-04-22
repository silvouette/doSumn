import util
import rank
from sklearn import metrics, svm
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

def try_knn(train, test): #classifying sentences to class 1 if summary-worthy, else 0
    x_train, y_train = train[['sentence_x','heading_x']], train[['abstract']]
    x_test, y_test = test[['sentence_x','heading_x']], test[['abstract']]

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
                    ('classifier', svm.SVC(class_weight='balanced', kernel='linear', C=10))
                    ])
    pipe.fit(x_train, y_train.values.ravel()) #data fitting
    y_pred = pipe.predict(x_test) #predict labels
    acc = metrics.accuracy_score(y_test, y_pred) #calculate accuracy
    accuracy = str(acc * 100) + '%' #accuracy to string for better printing

    return y_pred, accuracy

def classify(df_train, df_test): #classifying sentences into rhetorical roles [background, topic, method, dataset, result, conclusion, suggestion]
    x_train, y_train = df_train[['sentence','sentence_x','heading_x']], df_train[['labels']] #assigning train set
    x_test, y_test = df_test[['sentence','sentence_x','heading_x']], df_test[['labels']] #assigning test set

    transformer = FeatureUnion([ #TF-IDF
                ('sentence_tfidf', 
                  Pipeline([('extract_field',
                              FunctionTransformer(lambda x: x['sentence_x'], validate=False)),
                            ('tfidf', TfidfVectorizer(ngram_range=[1,3], max_features=2000))])),
                ('heading_tfidf', 
                  Pipeline([('extract_field', 
                              FunctionTransformer(lambda x: x['heading_x'], validate=False)),
                            ('tfidf', TfidfVectorizer())]))]) 
    pipe = Pipeline(steps=[ #Pipeline includes TF-IDF and classifier. Multiclass one-vs-rest SVM classifier used. Other parameters shown.
                    ('tfidf', transformer),
                    ('classifier', svm.SVC(class_weight='balanced', kernel='poly', decision_function_shape='ovr',  gamma=10, C=0.1))
                    ])
    pipe.fit(x_train, y_train.values.ravel()) #data fitting
    y_pred = pipe.predict(x_test) #predict labels
    acc = metrics.accuracy_score(y_test, y_pred) #calculate accuracy
    accuracy = str(acc * 100) + '%' #accuracy to string for better printing
    prediction = x_test
    prediction['labels'], prediction['pred'] = y_test, y_pred #new dataframe to include test expected value and predicted value

    result = rank.ranker(prediction) #rank sentences to filter out more sentences for shorter result
    return result, prediction, accuracy

def dosumn(filename):
    # data prep & clean
    train = util.get_parser_data("train_set.csv")
    test = util.get_parser_data("test/"+filename)

    for column in ["sentence","heading"]: #sentence cleaning includes lowercasing, removing extra parts and numbers, punctuations. Refer to util.py)
      train[column+"_x"] = train[column].apply(util.clean)
      test[column+"_x"] = test[column].apply(util.clean)

    # part 1: sentence removal, pick only summary-worthy sentence
    test['summary_worth'], rm_acc = try_knn(train, test)
    #part 2: labelling
    c_train = train[train['abstract'] == 1].copy()
    c_test = test[test['summary_worth'] == 1].copy()
    res, classed, cl_acc = classify(c_train, c_test) #classed is dataset to display in flask as result of 2nd classifiction. cl_acc is 2nd classification accuracy

    removal = test[['sentence_x','heading_x','labels','abstract','summary_worth']] #dataset to display in flask for first classification result   
    removal.rename(columns = {'labels' : 'r_role','abstract' : 'expected', 'summary_worth' : 'predicted'}, inplace = True) #renaming for better display
    classed.rename(columns = {'sentence' : 'sentence_ori', 'labels' : 'expected', 'pred' : 'predicted'}, inplace = True)
    word_count = util.wordcount(res) #wordcount to display in flask. refer to util.py

    return res, word_count, removal, classed, rm_acc, cl_acc

if __name__ == "__main__":  
    dosumn("class - 1.csv")