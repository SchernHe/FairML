"""Calculation of predictive parity - one sensitive attribute"""
import pandas as pd
from fairml.metrics.utils import calculate_precision


def calculate_predictive_parity(
    df, target_variable, prediction_variable, protected_variable_list, is_binary=True
):
    """
    Calculate the predictive parity metric for a given dataframe sensitive
    attribute

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
    predictive_paritiy: pd.DataFrame
    """

    predictive_parity = pd.DataFrame(
        columns=["Precision"], index=protected_variable_list + ["Gap"]
    )

    if not is_binary:
        # TO-DD
        return "Sensitive Attribute is not binary. Not supported at the moment"
    else:

        for protected_variable in protected_variable_list:

            df_prot_subset = df.loc[(df[protected_variable] == 1)]
            predictive_parity.loc[protected_variable] = calculate_precision(
                df_prot_subset, target_variable, prediction_variable
            )

        predictive_parity.loc["Gap", "Precision"] = (
            predictive_parity["Precision"].max() - predictive_parity["Precision"].min()
        )

        return predictive_parity
