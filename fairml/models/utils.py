import numpy as np
import tensorflow as tf
from fairml.metrics.fairness_metrics import calculate_fairness_metrics


def save_model(model, mean_gen_loss):
    model_json = model.generator.to_json()
    json_path = model.checkpoint_dir + "generator_" + str(mean_gen_loss) + ".json"
    hd5_path = model.checkpoint_dir + "generator_" + str(mean_gen_loss) + ".h5"
    with open(json_path, "w") as json_file:
        json_file.write(model_json)

    model.generator.save_weights(hd5_path)
    print("Saved model!")


def prepare_model_input(model, dataset):
    true_sensitive_values = dataset[model.sensitive_variables].values
    true_target_value = dataset[model.target_variable].values
    feature_cols = [
        col
        for col in dataset.columns
        if (col != model.target_variable) & (col not in model.sensitive_variables)
    ]
    feature_vector = dataset[feature_cols].values
    return feature_vector, true_sensitive_values, true_target_value


def prepare_generator_input(feature_vector, true_sensitive_values):
    generator_input = np.concatenate((feature_vector, true_sensitive_values), axis=1)
    return np.asmatrix(generator_input)


def prepare_discriminator_input(
    feature_vector, pred_target_value, true_target_value, disc_observation_based
):
    pred_target_value = tf.reshape(pred_target_value, (len(true_target_value), 1))
    if disc_observation_based:
        discriminator_input = np.concatenate(
            (feature_vector, pred_target_value), axis=1
        )
        return np.asmatrix(discriminator_input)
    else:
        return np.asmatrix(pred_target_value)


def save_fairness_metrics(model, dataset, fairness_series, epoch):
    feature_cols = [
        col
        for col in dataset.columns
        if (col != model.target_variable) & (col not in model.sensitive_variables)
    ]

    dataset["FairAN_y_score"] = model.generate_prediction(
        dataset[feature_cols + model.sensitive_variables]
    )
    dataset["FairAN_y"] = dataset["FairAN_y_score"].apply(
        lambda row: 1 if row > 0.5 else 0
    )

    FairAN_Fairness = calculate_fairness_metrics(
        dataset, model.target_variable, "FairAN_y", model.sensitive_variables
    )

    for key, value in fairness_series.items():
        if key == "Epoch":
            fairness_series.get(key).append(epoch)
        else:
            fairness_series.get(key).append(FairAN_Fairness.loc["Gap", key])

    return fairness_series


def init_placeholders():
    gen_loss_series = []
    disc_loss_series = []

    gen_loss_in_epoch = []
    disc_loss_in_epoch = []

    fairness_series = {
        "Epoch": [],
        "GroupFairness": [],
        "PredictiveParity": [],
        "TPR_EqOdds": [],
        "FPR_EqOdds": [],
    }

    return (
        gen_loss_series,
        disc_loss_series,
        gen_loss_in_epoch,
        disc_loss_in_epoch,
        fairness_series,
    )
