import numpy as np


def calculate_individual_fairness(
    model, df, columns, switch_binaries, is_benchmark=False
):

    if is_benchmark:
        df["Scores"] = model.predict_proba(df.iloc[:, columns])[:, 1]
        df_manipulated = _manipulate(df, switch_binaries)
        df["Scores_Manipulated"] = model.predict_proba(df_manipulated.iloc[:, columns])[
            :, 1
        ]
    else:
        df["Scores"] = model(df.iloc[:, columns].values).numpy()
        df_manipulated = _manipulate(df, switch_binaries)
        df["Scores_Manipulated"] = model(df_manipulated.iloc[:, columns].values).numpy()

    abs_mean_error = np.abs(df["Scores"] - df["Scores_Manipulated"]).mean()
    mean_sq_error = ((df["Scores"] - df["Scores_Manipulated"]) ** 2).mean()

    return abs_mean_error, mean_sq_error


def _manipulate(df, switch_binaries):
    df_manipulated = df.copy()

    for feature in switch_binaries:
        df_manipulated[feature] = df.apply(
            lambda row: _switch_feature_value(row, feature), axis=1
        )

    return df_manipulated


def _switch_feature_value(row, feature):
    if row[feature] == 1:
        return 0
    else:
        return 1
