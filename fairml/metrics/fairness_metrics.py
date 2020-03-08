import fairml.metrics.equalized_odds as equalized_odds
import fairml.metrics.statistical_parity as statistical_parity
import fairml.metrics.predictive_parity as predictive_parity


def calculate_fairness_metrics(
    df, target_variable, prediction_variable, protected_variable_list, is_binary=True
):
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

    sta_parity = statistical_parity.calculate_statistical_parity(
        df, prediction_variable, protected_variable_list
    )
    pre_parity = predictive_parity.calculate_predictive_parity(
        df, target_variable, prediction_variable, protected_variable_list
    )
    eq_odds = equalized_odds.calculate_equalized_odds(
        df, target_variable, prediction_variable, protected_variable_list
    )

    fairness_metrics = sta_parity.merge(
        pre_parity, left_index=True, right_index=True
    ).merge(eq_odds, left_index=True, right_index=True)
    return fairness_metrics
