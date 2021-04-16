import os
import numpy as np

def validate(x_test,y_test,y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    fn, fp = [], []

    for i in range(len(y_pred)):
        if y_test['abstract'].iloc[i] == 1:
            if y_pred[i] == 1:
                TP += 1
            else:
                FN += 1
                fn.append(x_test['sentence'].iloc[i])
        else:
            if y_pred[i] == 0:
                TN += 1
            else:
                FP += 1
                fp.append(x_test['sentence'].iloc[i])
    
    out = open("fpc.txt","w+")
    out.write("\n".join(fp))
    out.close()
                
    out = open("fnc.txt","w+")
    out.write("\n".join(fn))
    out.close()
                
    return TP, TN, FP, FN

def validate_more(data):
    bg, tp, mt, dt, rs, cc, sg = 0, 0, 0, 0, 0, 0, 0
    abg, atp, amt, adt, ars, acc, asg= [],[],[],[],[],[],[]

    for index, row in data.iterrows():
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
        
        
    print(len(abg), len(atp), len(amt), len(adt), len(ars), len(acc), len(asg))
    print("\nabg\n", abg)
    print("\natp\n", atp)
    print("\namt\n", amt)
    print("\nadt\n", adt)
    print("\nars\n", ars)
    print("\nacc\n", acc)
    print("\nasg\n", asg)
    # return bg, tp, mt, dt, rs, cc, sg, no