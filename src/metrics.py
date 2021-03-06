import matplotlib.pyplot as plt
import numpy as np
from icecream import ic 
import seaborn as sns
import pandas as pd
from sklearn import metrics
import cv2
import pdb


def plotAUC(fpr, tpr, roc_auc, modelId = '', nameId = ''):
    
    plt.plot([0, 1], [0, 1], 'k--')
    # roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
    # plt.plot(fpr, tpr, marker='.', label = 'AUC = %0.2f' % roc_auc)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(modelId + '. AUC = %0.2f' % roc_auc)
    plt.savefig('roc_auc_'+modelId+'_.png', dpi = 500)

def getBestThresholdGMean(fpr, tpr, thresholds):
    # calculate the g-mean for each threshold
    '''
    calculates the geometric mean for each threshold.
    Finds threshold with largest geometric mean.
    '''
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    best_threshold = thresholds[ix]
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    return ix, best_threshold

def getBestThresholdJStatistic(fpr, tpr, thresholds):
    # calculate the g-mean for each threshold
    J = tpr - fpr
    ix = np.argmax(J)
    best_threshold = thresholds[ix]
    print('Best Threshold=%f' % (best_threshold))
    return ix, best_threshold



def plotBestThreshold(fpr, tpr, ix):
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')

def plotHistogram(y_test, y_pred_score, plot_lims):
    '''
    print(np.unique(y_pred, return_counts=True),
        np.unique(y_test, return_counts=True))
    correct_idxs = np.equal(y_pred, y_test)
    print(np.unique(correct_idxs, return_counts=True))
    '''

    y_pred_correct = y_pred_score[y_test == 0]
    y_pred_incorrect = y_pred_score[y_test == 1]
    
    # plt.hold(True)
    '''
    fig, axs = plt.subplots(2)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    axs[0].hist(y_pred_correct, 100, density=False, facecolor='g', alpha=0.5)
    axs[0].axis(xmin = plot_lims[0][0], xmax = plot_lims[0][1])
    # plt.figure()
    axs[1].hist(y_pred_incorrect, 100, density=False, facecolor='b', alpha=0.5)
    axs[1].axis(xmin = plot_lims[0][0], xmax = plot_lims[0][1])
    '''

    fig, axs = plt.subplots(1)
    fig.set_figheight(7)
    fig.set_figwidth(12)
    axs.hist(y_pred_correct, 300, ls='dashed', density=False, facecolor='g', alpha=0.5, label="Correct")
    axs.axis(xmin = plot_lims[0][0], xmax = plot_lims[0][1],
            ymax = plot_lims[1][1])
    axs.hist(y_pred_incorrect, 300, ls='dashed', density=False, facecolor='b', alpha=0.5, label="Incorrect")
    axs.axis(xmin = plot_lims[0][0], xmax = plot_lims[0][1],
            ymax = plot_lims[1][1])
    plt.legend(loc="upper right")
    plt.xlabel('Uncertainty')
    plt.ylabel('Sample count')

    '''
    sns.histplot(data=pd.DataFrame({"uncertainty": y_pred_score,
        "correct": correct_idxs,
        }),
    x="uncertainty", hue="correct", kde=False)
    '''

def getTprFprFromConfusionMatrix(cm):

    _tn, _fp, _fn, _tp = cm.ravel()
    ic(_tn, _fp, _fn, _tp)

    tpr = _tp/(_tp+_fn)
    fpr = _fp/(_fp+_tn)
    return tpr, fpr

def plotConfusionMatrix(cm, target_names = ['correct', 'incorrect']):
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(3,3))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # plt.show(block=False)


# =================
# 4 classes error detection


def createErrorMaskPerClass(label,
        predicted, mask_values):
    '''
    Creates error mask per class
    0: correct class 0.
    1: correct class 1.
    2: incorrect class 0.
    3: incorrect class 1.
    '''
    len = label.shape[0]
    error_mask_per_class = np.zeros_like(label)
    for idx in range(len):
        if label[idx] == 0 and predicted[idx] == 0:
            error_mask_per_class[idx] = mask_values[0]
        elif label[idx] == 0 and predicted[idx] == 1:
            error_mask_per_class[idx] = mask_values[1]        
        elif label[idx] == 1 and predicted[idx] == 0:
            error_mask_per_class[idx] = mask_values[2]
        elif label[idx] == 1 and predicted[idx] == 1:
            error_mask_per_class[idx] = mask_values[3]

    return error_mask_per_class

def getConfusionMatrix4ClassesReduced(cm_4classes):
    cm_4classes_reduced = np.zeros((4,2))
    for row in range(4):
        for col in range(2):
            # 0: 0,1 . 1: 2,3
            cm_4classes_reduced[row, col] = cm_4classes[row, col*2] + cm_4classes[row, col*2 + 1] 

    cm_4classes_reduced_percentage = np.zeros_like(cm_4classes_reduced)

    for row in range(4):
        _sum = cm_4classes_reduced[row, 0] + cm_4classes_reduced[row, 1]
        for col in range(2):
            cm_4classes_reduced_percentage[row, col] = round((cm_4classes_reduced[row, col] / _sum)*100, 1)

    return cm_4classes_reduced, cm_4classes_reduced_percentage

def getErrorConfusionMatrixFromPredictedAsClassX(error_mask_test, 
    y_pred_thresholded, predicted_test, x):
    y_pred_thresholded_classX = y_pred_thresholded[predicted_test == x]
    error_mask_test_classX = error_mask_test[predicted_test == x]

    cm_classX = metrics.confusion_matrix(error_mask_test_classX, y_pred_thresholded_classX)
    print(cm_classX)

    plotConfusionMatrix(cm_classX)

# =====================
# precision-recall curve

def plotPrecisionRecall(label, precision, recall, ix):
    no_skill = len(label[label==1]) / len(label)
    plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, label='ResUnet')
    plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall curve')
    plt.legend()

def getRgbErrorMask(predicted, label):
    false_positive_mask = predicted - label

    error_mask_to_show = predicted.copy()
    error_mask_to_show[false_positive_mask == 1] = 2
    error_mask_to_show[false_positive_mask == -1] = 3
    return error_mask_to_show
def saveRgbErrorMask(error_mask_to_show, dim = None):


    colormap = np.array([[0, 0, 0],
            [255, 255, 255],
            [0, 0, 255],
            [255, 0, 0]])

    colormap = np.array([[255, 255, 255],
            [0, 0, 0],
            [45, 150, 255],
            [255, 146, 36]])


    
    error_mask_to_show_rgb=cv2.cvtColor(error_mask_to_show,cv2.COLOR_GRAY2RGB)
    error_mask_to_show_rgb_tmp = error_mask_to_show_rgb.copy()
    for idx in range(colormap.shape[0]):
        for chan in range(error_mask_to_show_rgb.shape[-1]):
            error_mask_to_show_rgb[...,chan][error_mask_to_show_rgb_tmp[...,chan] == idx]=colormap[idx,chan]

    error_mask_to_show_rgb=cv2.cvtColor(error_mask_to_show_rgb,cv2.COLOR_BGR2RGB)
    if dim is not None:
        error_mask_to_show_rgb = cv2.resize(error_mask_to_show_rgb, 
            dim, interpolation = cv2.INTER_NEAREST)
    return error_mask_to_show_rgb

def getAA_Recall(predict_probability, label_mask_current_deforestation_test, 
        predicted_test, threshold_list):
    metrics_list = []
    for threshold in threshold_list:
        print("threshold", threshold)
        predicted_thresholded = np.zeros_like(predict_probability).astype(np.int8)
        predicted_thresholded[predict_probability >= threshold] = 1

        predicted_test_classified_correct = predicted_test[
            predicted_thresholded == 0]
        label_current_deforestation_test_classified_correct = label_mask_current_deforestation_test[
            predicted_thresholded == 0]

        predicted_test_classified_incorrect = predicted_test[
            predicted_thresholded == 1]
        label_current_deforestation_test_classified_incorrect = label_mask_current_deforestation_test[
            predicted_thresholded == 1]

        print(label_current_deforestation_test_classified_correct.shape,
            predicted_test_classified_correct.shape)
        cm_correct = metrics.confusion_matrix(
            label_current_deforestation_test_classified_correct,
            predicted_test_classified_correct)
        print("cm_correct", cm_correct)

        TN_L = cm_correct[0,0]
        FN_L = cm_correct[1,0]
        TP_L = cm_correct[1,1]
        FP_L = cm_correct[0,1]

        print(label_current_deforestation_test_classified_incorrect.shape,
            predicted_test_classified_incorrect.shape)

        cm_incorrect = metrics.confusion_matrix(
            label_current_deforestation_test_classified_incorrect,
            predicted_test_classified_incorrect)

        print("cm_incorrect", cm_incorrect)
        TN_H = cm_incorrect[0,0]
        FN_H = cm_incorrect[1,0]
        TP_H = cm_incorrect[1,1]
        FP_H = cm_incorrect[0,1]
        
        precision_L = TP_L / (TP_L + FP_L)
        recall_L = TP_L / (TP_L + FN_L)
        
        precision_H = TP_H / (TP_H + FP_H)
        recall_H = TP_H / (TP_H + FN_H)
        
        recall_Ltotal = TP_L / (TP_L + FN_L + TP_H + FN_H)
        AA = (TP_H + FN_H + FP_H + TN_H) / len(label_mask_current_deforestation_test)
        
        mm = np.hstack((precision_L, recall_L, recall_Ltotal, AA,
                precision_H, recall_H))
        print(mm)
        metrics_list.append(mm)

        # pdb.set_trace()
    metrics_list = np.asarray(metrics_list)
    return metrics_list       


def getF1byThreshold(score, label, threshold_list):
    metrics_list = []
    for threshold in threshold_list:
        print("threshold", threshold)
        predicted_thresholded = np.zeros_like(score).astype(np.int8)
        predicted_thresholded[score >= threshold] = 1
        cm = metrics.confusion_matrix(
            label,
            predicted_thresholded)
        print("cm", cm_correct)

        TN = cm_correct[0,0]
        FN = cm_correct[1,0]
        TP = cm_correct[1,1]
        FP = cm_correct[0,1]
        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)

        
        mm = np.hstack((precision_L, recall_L, recall_Ltotal, AA,
                precision_H, recall_H))
        print(mm)
        metrics_list.append(mm)

        # pdb.set_trace()
    metrics_list = np.asarray(metrics_list)
    return metrics_list       

'''
def matrics_AA_recall(thresholds_, prob_map, ref_reconstructed, mask_amazon_ts_, px_area):
    thresholds = thresholds_    
    metrics_all = []
    
    for thr in thresholds:
        print(thr)  

        img_reconstructed = np.zeros_like(prob_map).astype(np.int8)
        img_reconstructed[prob_map >= thr] = 1
    
        mask_areas_pred = np.ones_like(ref_reconstructed)
        area = skimage.morphology.area_opening(img_reconstructed, area_threshold = px_area, connectivity=1)
        area_no_consider = img_reconstructed-area
        mask_areas_pred[area_no_consider==1] = 0
        
        # Mask areas no considered reference
        mask_borders = np.ones_like(img_reconstructed)
        #ref_no_consid = np.zeros((ref_reconstructed.shape))
        mask_borders[ref_reconstructed==2] = 0
        #mask_borders[ref_reconstructed==-1] = 0
        
        mask_no_consider = mask_areas_pred * mask_borders 
        ref_consider = mask_no_consider * ref_reconstructed
        pred_consider = mask_no_consider*img_reconstructed
        
        ref_final = ref_consider[mask_amazon_ts_==1]
        pre_final = pred_consider[mask_amazon_ts_==1]
        
        # Metrics
        cm = confusion_matrix(ref_final, pre_final)
        #TN = cm[0,0]
        FN = cm[1,0]
        TP = cm[1,1]
        FP = cm[0,1]
        precision_ = TP/(TP+FP)
        recall_ = TP/(TP+FN)
        mm = np.hstack((recall_, precision_))
        metrics_all.append(mm)
    metrics_ = np.asarray(metrics_all)
    return metrics_
'''