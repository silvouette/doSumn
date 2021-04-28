import util
import rank
import os
import pandas as pd
from sklearn import metrics, svm
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

# test = util.get_parser_data("test_set.csv")
test = util.get_parser_data("test/"+"class - 17.csv")
for column in ["sentence","heading"]: #sentence cleaning includes lowercasing, removing extra parts and numbers, punctuations. Refer to util.py)
  test[column+"_x"] = test[column].apply(util.clean)

x_test, y_test = test[['sentence_x','heading_x']], test[['abstract']]

print(x_test['sentence_x'])