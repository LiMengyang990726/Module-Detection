def confusionMatrix(TP,FN,FP,TN):
    F1 = (2*TP)/(2*TP+FP+FN)
    Accuracy = (TP+TN) / (TP+FN+FP+TN)
    Recall = (TP) / (TP+FN)
    Precision = (TP) / (TP+FP)
    result = []
    result.append(Accuracy)
    result.append(Precision)
    result.append(Recall)
    result.append(F1)
    return (result)
