"""Calculation of group fairness - one sensitive attribute"""
import pandas as pd
import numpy as np 


def calculate_group_fairness(df, prediction_variable, protected_variable_list, is_binary = True):
	"""
	Calculate the group fairness metric for a given dataframe sensitive attribute
	
	Parameters
	----------
	df : pd.DataFrame
	prediction_variable : str
		Binary target variable (0/1)
	protected_variable_list : list
		List of the one hot encoded protected variable
	is_binary : bool, optional
	    Flag whether sensitive attribute is binary or not
	
	Returns
	-------
	group_fairness : pd.DataFrame
	"""

	group_fairness = pd.DataFrame(
		columns=["GroupFairness"],
		index=protected_variable_list + ["Gap"] 
		)

	if not is_binary:
		# TO-DD
		return "Sensitive Attribute is not binary. Not supported at the moment"
	else:

		for protected_variable in protected_variable_list:
			df_prot_subset = df.loc[df[protected_variable]==1]
			num_of_instances = len(df_prot_subset)
			num_of_positive_classification = len(df_prot_subset.loc[df_prot_subset[prediction_variable]==1])
			group_fairness.loc[protected_variable] = (num_of_positive_classification/num_of_instances)*100 
		
		group_fairness.loc["Gap","GroupFairness"] = group_fairness["GroupFairness"].max() - group_fairness["GroupFairness"].min()
		
		return group_fairness