"""Provides useful functions to calculate fairness metrics"""
import pandas as pd 
from sklearn.metrics import confusion_matrix

def calculate_precision(df, target_variable, prediction_variable):
	"""Calculate precision / positive predictive value PPV"""
	tn, fp, fn, tp = confusion_matrix(df[target_variable], df[prediction_variable]).ravel()
	return (tp/(tp+fp))*100


def calculate_recall(df, target_variable, prediction_variable):
	"""Calculate recall / true positive rate TPR / sensitivity"""
	tn, fp, fn, tp = confusion_matrix(df[target_variable], df[prediction_variable]).ravel()
	return (tp/(tp+fn))*100


def calculate_fpr(df, target_variable, prediction_variable):
	"""Calculate false positive rate FPR / false alarm ratio"""
	tn, fp, fn, tp = confusion_matrix(df[target_variable], df[prediction_variable]).ravel()
	return (fp/(fp+tn))*100