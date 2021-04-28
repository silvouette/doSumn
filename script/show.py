import util
import rank
import os
import pandas as pd
from sklearn import metrics

path = os.path.dirname(os.getcwd())
rm_path = os.path.join(path, "json_rm/")
cl_path = os.path.join(path, "json_cl/")

def dosumn(filename):

    name = os.path.splitext(filename)[0]
    rm = pd.read_json(rm_path+name+'.json')
    cl = pd.read_json(cl_path+name+'.json')
    
    y_test, y_pred = rm['abstract'], rm['summary_worth']
    rm_acc = metrics.accuracy_score(y_test, y_pred)*100
    y_test, y_pred = cl['labels'], cl['pred']
    cl_acc = metrics.accuracy_score(y_test, y_pred)*100

    res = rank.ranker(cl)

    print(cl.shape)
    
    removal = rm[['sentence_x','heading_x','labels','abstract','summary_worth']] #dataset to display in flask for first classification result   
    removal.rename(columns = {'labels' : 'r_role','abstract' : 'expected', 'summary_worth' : 'predicted'}, inplace = True) #renaming for better display
    cl.rename(columns = {'sentence' : 'sentence_ori', 'labels' : 'expected', 'pred' : 'predicted'}, inplace = True)
    word_count = util.wordcount(res) #wordcount to display in flask. refer to util.py

    return res, word_count, removal, cl, rm_acc, cl_acc

if __name__ == "__main__":  

  dosumn("class - 1.csv")
