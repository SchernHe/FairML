"""Calculation of group fairness - one sensitive attribute"""
import pandas as pd


def calculate_statistical_parity(
    df, prediction_variable, protected_variable_list, is_binary=True
):
    """
    Calculate statistical parity for a given dataframe sensitive attribute

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
    statistical_parity : pd.DataFrame
    """

    statistical_parity = pd.DataFrame(
        columns=["Acceptance_Rate"], index=protected_variable_list + ["Gap"]
    )

    if not is_binary:
        # TO-DD
        return "Sensitive Attribute is not binary. Not supported at the moment"
    else:

        for protected_variable in protected_variable_list:
            df_prot_subset = df.loc[df[protected_variable] == 1]
            num_of_instances = len(df_prot_subset)
            num_of_positive_classification = len(
                df_prot_subset.loc[df_prot_subset[prediction_variable] == 1]
            )
            statistical_parity.loc[protected_variable] = (
                num_of_positive_classification / num_of_instances
            ) * 100

        statistical_parity.loc["Gap", "Acceptance_Rate"] = (
            statistical_parity["Acceptance_Rate"].max()
            - statistical_parity["Acceptance_Rate"].min()
        )

        return statistical_parity
