import numpy as np
import re
import rank
from collections import defaultdict

def validate(x_test,y_test,y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    fn, fp = [], []

    for i in range(len(y_pred)):
        if y_test['abstract'].iloc[i] == 1:
            if y_pred[i] == 1:
                TP += 1
            else:
                FN += 1
                fn.append(x_test['sentence_x'].iloc[i])
        else:
            if y_pred[i] == 0:
                TN += 1
            else:
                FP += 1
                fp.append(x_test['sentence_x'].iloc[i])        
    return TP, TN, FP, FN

def validate_more(data):
    # regex = re.compile(r',\s*\d{4}\)')
    # atp = [i for i in atp if not regex.search(i)]
    sets = ['BACKGROUND','TOPIC','METHOD','DATASET','RESULT','CONCLUSION','SUGGESTION']
    res = []
    summ_collection = {}
    for cat in sets:
        mask = data.loc[data['pred']== cat]
        if len(mask)>=10:
            summ = rank.generate_summary(data.loc[data['pred']== cat], int(len(mask)*0.4))
        elif len(mask)>=2:
            summ = rank.generate_summary(data.loc[data['pred']== cat], int(len(mask)*0.5))
        else:
            summ = data.loc[data['pred']== cat,'sentence'].values

        res.append(summ)
        summ_collection[cat] = summ

    return summ_collection


