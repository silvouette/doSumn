import os
import numpy as np
import re

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
    abg, atp, amt, adt, ars, acc, asg= [],[],[],[],[],[],[]

    for index,row in data.iterrows():
        if row['pred'] == 'BACKGROUND':
            abg.append(row['sentence'])
        elif row['pred'] == 'TOPIC':
            atp.append(row['sentence'])
        elif row['pred'] == 'METHOD':
            amt.append(row['sentence'])
        elif row['pred'] == 'DATASET':
            adt.append(row['sentence'])
        elif row['pred'] == 'RESULT':
            ars.append(row['sentence'])
        elif row['pred'] == 'CONCLUSION':
            acc.append(row['sentence'])
        else:
            asg.append(row['sentence'])
    
    regex = re.compile(r',\s*\d{4}\)')
    atp = [i for i in atp if not regex.search(i)]
      
    # print(abg,atp,amt,adt,ars,acc,asg)

    arr = [abg,atp,amt,adt,ars,acc,asg]
    
    for i in arr:
        if len(i) > 5:
            print("summ!")
