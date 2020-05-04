import numpy as np


def calculate_discrepancy(df, informative_variables, Y_model, Y_benchmark):
    benchmark_metric = 0
    model_metric = 0
    
    grouped = df.groupby(informative_variables)
    count_groups = len(grouped.groups)

    for id, group in grouped:

        if not np.isnan(group[Y_benchmark].std()):
            benchmark_metric += group[Y_benchmark].std()

        if not np.isnan(group[Y_model].std()):
            model_metric += group[Y_model].std()

    model_discrepancy = model_metric / count_groups
    benchmark_discrepancy = benchmark_metric / count_groups

    return model_discrepancy, benchmark_discrepancy
