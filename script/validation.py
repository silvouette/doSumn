import os

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
    
    # out = open("fpc.txt","w+")
    # out.write("\n".join(fp))
    # out.close()
                
    return TP, TN, FP, FN