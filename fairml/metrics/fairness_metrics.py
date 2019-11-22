from fairml.metrics import *

def calculate_fairness_metrics(df, target_variable, prediction_variable, protected_variable_list, is_binary = True):
	"""Calculate available fairness metrics to give overview over the given data
	
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
	fairness_metrics: pd.DataFrame
	"""
	
	gf_metrics = group_fairness.calculate_group_fairness(df, prediction_variable, protected_variable_list)
	pp_metrics = predictive_parity.calculate_predictive_parity(df, target_variable, prediction_variable, protected_variable_list)
	eo_metrics = equalized_odds.calculate_equalized_odds(df, target_variable, prediction_variable, protected_variable_list)

	fairness_metrics = gf_metrics.merge(pp_metrics,left_index=True,right_index=True).merge(eo_metrics,left_index=True,right_index=True)
	return fairness_metrics