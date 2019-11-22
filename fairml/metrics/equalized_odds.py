"""Calculation of predictive parity - one sensitive attribute"""
import pandas as pd
import numpy as np 
from fairml.metrics.utils import calculate_recall, calculate_fpr

def calculate_equalized_odds(df, target_variable, prediction_variable, protected_variable_list, is_binary = True):
	"""
	Calculate the group fairness metric for a given dataframe sensitive attribute
	
	Parameters
	----------
	df : pd.DataFrame
	target_variable: str
	prediction_variable : str
		Binary target variable (0/1)
	protected_variable_list : list
		List of the one hot encoded protected variable
	is_binary : bool, optional
	    Flag whether sensitive attribute is binary or not
	
	Returns
	-------
	equalized_odds: pd.DataFrame
	"""

	equalized_odds = pd.DataFrame(
		columns=["TPR_EqOdds","FPR_EqOdds"],
		index=protected_variable_list + ["Gap"] 
		)

	if not is_binary:
		# TO-DD
		return "Sensitive Attribute is not binary. Not supported at the moment"
	else:

		for protected_variable in protected_variable_list:
			df_prot_subset = df.loc[(df[protected_variable]==1)]
			equalized_odds.loc[protected_variable,"TPR_EqOdds"] = calculate_recall(df_prot_subset, target_variable, prediction_variable)
			equalized_odds.loc[protected_variable,"FPR_EqOdds"] = calculate_fpr(df_prot_subset, target_variable, prediction_variable)
		equalized_odds.loc["Gap","TPR_EqOdds"] = equalized_odds["TPR_EqOdds"].max() - equalized_odds["TPR_EqOdds"].min()
		equalized_odds.loc["Gap","FPR_EqOdds"] = equalized_odds["FPR_EqOdds"].max() - equalized_odds["FPR_EqOdds"].min()

		return equalized_odds