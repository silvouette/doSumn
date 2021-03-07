import pandas as pd
import nltk
import string
import re
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
pd.options.mode.chained_assignment = None

def get_parser_data(filename):
    data = pd.read_csv('../' + filename)
    return data

def preprocessing(text):
    noise = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'yang', 'akan', 'dan']
    text = text.translate(str.maketrans("","",string.punctuation)).strip().lower()
    text = re.sub(r"\d+", "", text)
    text = ' '.join(w for w in text.split() if w not in noise)
    tokens = nltk.tokenize.word_tokenize(text)
    return tokens

def try_knn():
    df = get_parser_data("dataset.csv")
    x = df[['sentence','heading']]
    y = df[['abstract']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    for column in ["sentence","heading"]:
        x_train[column] = x_train[column].apply(preprocessing)
        x_test[column] = x_test[column].apply(preprocessing)
    
    print(x_train,x_test)
if __name__ == "__main__":  
    try_knn()

