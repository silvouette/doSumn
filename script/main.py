import util
import rank
from sklearn import metrics, svm
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

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
    acc = metrics.accuracy_score(y_test, y_pred)
    accuracy = str(acc * 100) + '%'

    return y_pred, accuracy

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
    accuracy = str(acc * 100) + '%'
    prediction = x_test
    prediction['labels'], prediction['pred'] = y_test, y_pred

    result = rank.ranker(prediction)
    return result, prediction, accuracy

def dosumn(filename):
    # data prep & clean
    train = util.get_parser_data("train_set.csv")
    test = util.get_parser_data("test/"+filename)

    for column in ["sentence","heading"]:
      train[column+"_x"] = train[column].apply(util.clean)
      test[column+"_x"] = test[column].apply(util.clean)

    # part 1: sentence removal, pick only summary-worthy sentence
    test['summary_worth'], rm_acc = try_knn(train, test)
    #part 2: labelling
    c_train = train[train['abstract'] == 1].copy()
    c_test = test[test['summary_worth'] == 1].copy()
    res, classed, cl_acc = classify(c_train, c_test) #here

    removal = test[['sentence_x','heading_x','labels','abstract','summary_worth']]    
    word_count = util.wordcount(res)

    return res, word_count, removal, classed, rm_acc, cl_acc

if __name__ == "__main__":  
    dosumn("class - 1.csv")