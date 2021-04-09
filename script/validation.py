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

def validate_more(x_test,y_test,y_pred):
    bg, tp, mt, dt, rs, cc, sg, no = 0, 0, 0, 0, 0, 0, 0, 0
    # fn, fp = [], []

    for i in range(len(y_pred)):
        if y_pred[i][0] == 1:
            bg += 1
            # print('background')
        if y_pred[i][1] == 1:
            cc += 1
            # print('conclusion')
        if y_pred[i][2] == 1:
            dt += 1
            # print('dataset')
        if y_pred[i][3] == 1:
            mt += 1
            # print('method')
        if y_pred[i][4] == 1:
            rs += 1
            # print('result')
        if y_pred[i][5] == 1:
            sg += 1
            # print('suggestion')   
        if y_pred[i][6] == 1:
            tp += 1
            # print('topic')
        if np.array_equal(y_pred[i], [0,0,0,0,0,0,0,0]):
            no += 1
            # print('NOTHING') 
        print(y_pred[i])         
    return bg, tp, mt, dt, rs, cc, sg, no