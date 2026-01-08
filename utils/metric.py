import numpy as np

def get_metric(confusion_matrix):
    TP = confusion_matrix[0]
    FP = confusion_matrix[1]
    TN = confusion_matrix[2]
    FN = confusion_matrix[3]

    # 计算准确度（Accuracy）
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # 计算精确度（Precision）
    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0.0

    # 计算召回率（Recall）
    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0

    # 计算F1分数
    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    # 计算IoU（Intersection over Union）
    iou = TP / (TP + FP + FN)

    return accuracy, f1_score, iou, precision, recall

def get_confusion_matrix(predicted_labels,true_labels):
    true_labels = (true_labels / true_labels.max()).astype(float)
    predicted_labels = (predicted_labels / predicted_labels.max()).astype(float)
    predicted_labels = predicted_labels.astype('uint8')
    true_labels = true_labels.astype('uint8')


    TP = np.sum(np.logical_and(true_labels == 1, predicted_labels == 1))
    FP = np.sum(np.logical_and(true_labels == 0, predicted_labels == 1))
    TN = np.sum(np.logical_and(true_labels == 0, predicted_labels == 0))
    FN = np.sum(np.logical_and(true_labels == 1, predicted_labels == 0))

    return [TP, FP, TN, FN]