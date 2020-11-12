#!/usr/bin/env python
# coding: utf-8

# ### Calculate the Model's accuracy and prediction rate
# import pandas as pd
import numpy as np

# get the TN value based on TP, FP, FN and image shape
def get_TN(tot_pixels, TP, FP, FN):
    TN = tot_pixels - TP - FP - FN
    return TN


# precision = correctly identified objects / identified objects
# area of the correctly detected house / area of the detected house
# TP + FP = 95141.74
def model_precision(TP, FP):
    pre_rate = round((TP / (TP + FP)) * 100, 4)
    print("The model's Precision is: "+'{:.2f}'.format(pre_rate) + "%.")
    return pre_rate


# recall reate = correctly identified objects / total of the objects
# area of the detected house / area of the taged house
# TP + FN = 102097.565412
def model_recall(TP, FN):
    reca_rate = round((TP / (TP + FN))* 100, 4)
    print("The model's Recall is: "+'{:.2f}'.format(reca_rate)+ "%.")
    return reca_rate



# Accuracy = TP + TN / (TP + FP + TN + FN)
def model_accuracy(TP, TN, tot_pixels):
    accu_rate = round((TP +TN )/ tot_pixels * 100, 4)
    print("The model's Accuracy is: "+'{:.2f}'.format(accu_rate)+ "%.")
    return accu_rate


# return confusion matrix
def confusion_matrix(TP, TN, FP, FN):
    '''
    col_name = ["True Structure", "True Background", "Predict Total"]
    row_name = ["Predict Structure", "Predict Background", "True Total"]
    
    cf_matrix = np.array([[TP, FP, TP + FP],
                          [FN, TN, FN + TN],
                          [TP + FN, FP + TN, TP + FP + TN + FN]])
    '''
    
    
    cf_matrix = np.array([[" ","True Structure", "True Background", "Predict Total", "Unit"],
                          ["Predict Structure",TP, 
                           FP, TP + FP, '1pixel'],
                          ["Predict Background",FN, 
                           TN,(FN + TN), '1pixel'],
                          ["True Total",(TP + FN), (FP + TN), 
                           (TP + FP + TN + FN), '1pixel']])
    
    # df = pd.DataFrame(cf_matrix, row_name, col_name)
    # print(format_matrix(df))    
    return cf_matrix 


# Specificity= TN/(TN+FP) = TN/(Actual No)
def specificity_rate(TN, FP):
    spec_rate = round(TN / (TN + FP) * 100, 4)
    print("The model's Specificity is: "+'{:.2f}'.format(spec_rate)+ "%.")
    return spec_rate
    
# Negative Predictive = TN/(TN + FN)
def neg_predictive_rate(TN, FP):
    negpred_rate = round(TN / (TN + FN) * 100, 4)
    print("The model's Negative Predictive is: "+'{:.2f}'.format(negpred_rate)+ "%.")
    return negpred_rate    
    
    
    
# Prevalence: How often does the yes condition actually occur in our sample?
# Prevalence= Actual yes/Total = ( FN+TP)/(TP+FP+TN+FN) 
def prevalence_rate(FN, TP, tot_pixels):
    prev_rate = round((FN + TP )/ tot_pixels * 100, 4)
    print("The model's Prevalence is: "+'{:.2f}'.format(prev_rate)+ "%.")
    return prev_rate
    
    
# Cohen's Kappa
# input is the confusion matrix
'''
def kappa_rate(cf_matrix):        
    NM = cf_matrix.iloc[0,2] * cf_matrix.iloc[2,0]
    GC = cf_matrix.iloc[1,2] * cf_matrix.iloc[2,1]
    NN = cf_matrix.iloc[2,2] * cf_matrix.iloc[2,2]
    pe = (NM + GC) / NN * 1.0
    p0 = (cf_matrix.iloc[0,0] + cf_matrix.iloc[1,1]) / cf_matrix.iloc[2,2]
    kappa_rate = ((p0 - pe) / (1 - pe)) * 100
    print("The kappa is: "+'{:.2f}'.format(kappa_rate)+ "%.")
    return round(kappa_rate, 4)
 

# format output matrix
def format_matrix(df):
    new_df = df.div(100000)
    new_matrix = new_df.round(1)
    new_matrix['Unit'] = '100K'
    return new_matrix
'''

def format_matrix(cf_matrix):
    cf_matrix[1][1] = round(float(cf_matrix[1][1])/100000, 1)
    cf_matrix[1][2] = round(float(cf_matrix[1][2])/100000, 1)
    cf_matrix[1][3] = round(float(cf_matrix[1][3])/100000, 1)
    cf_matrix[2][1] = round(float(cf_matrix[2][1])/100000, 1)
    cf_matrix[2][2] = round(float(cf_matrix[2][2])/100000, 1)
    cf_matrix[2][3] = round(float(cf_matrix[2][3])/100000, 1)
    cf_matrix[3][1] = round(float(cf_matrix[3][1])/100000, 1)
    cf_matrix[3][2] = round(float(cf_matrix[3][2])/100000, 1)
    cf_matrix[3][3] = round(float(cf_matrix[3][3])/100000, 1)
    cf_matrix[1][4] = '100k'
    cf_matrix[2][4] = '100k'
    cf_matrix[3][4] = '100k'
    return cf_matrix
    
    

# F1 score = 2*(Precison * Recall)/ (Precision + Recall)
def F1_score(TP, FP, FN):
    recall = model_recall(TP, FN)
    precision = model_precision(TP, FP)
    F1_score = 2 * precision * recall / (precision + recall)
    print("The F1 is: "+'{:.2f}'.format(F1_score)+ "%.")
    return F1_score


# balanced accuracy=sensitivity + specificity/2
def bal_accuracy(TP, TN, FN, FP):
    sensi_rate = model_recall(TP, FN)
    speci_rate = specificity_rate(TN, FP)
    bal_accuracy = 0.5 * (sensi_rate + speci_rate)
    print("The Balanced Accuracy is: "+'{:.2f}'.format(bal_accuracy)+ "%.")
    return bal_accuracy