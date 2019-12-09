"""Deep Adversarial Network to mitigate bias

Components
----------
    Generator: Tries to predict the target value while tricking the discriminator
                y_hat = Prob( y=1 | x, s)

    Discriminator: Tries to predict a sensitive attribute based on
        * the observation (all attributes plus prediction of Generator)
                s_hat = Prob( s=1 | x , y_hat)
        * the prediction (without any additional attributes)
                s_hat = Prob( s=1 | y_hat)
Loss-Function
-------------
    We use the binary-cross-entropy for both the discriminator and generator network
    for the optimazation.

"""

import tensorflow as tf
from tensorflow.keras import layers
from fairml.models.utils import (
    save_model,
    save_fairness_metrics,
    prepare_model_input,
    prepare_generator_input,
    prepare_discriminator_input,
)
import time
import numpy as np


class FairAN:
    def __init__(
        self,
        generator_optimizer,
        discriminator_optimizer,
        sensitive_variables,
        target_variable,
        checkpoint_dir,
        fairness_multiplier,
        save_fairness_in,
        disc_observation_based,
    ):
        tf.keras.backend.set_floatx("float64")
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.sensitive_variables = sensitive_variables
        self.target_variable = target_variable
        self.checkpoint_dir = checkpoint_dir
        self.fairness_multiplier = fairness_multiplier
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.save_fairness_in = save_fairness_in
        self.disc_observation_based = disc_observation_based

    def make_generator(
        self, num_neurons: int, num_layers: int, input_shape: (int, None)
    ):
        """Create Generator Network"""

        model = tf.keras.Sequential()
        model.add(
            layers.Dense(
                num_neurons,
                input_shape=input_shape,
                kernel_initializer="glorot_normal",
                activation="relu",
            )
        )

        for i in range(num_layers - 1):
            model.add(
                layers.Dense(
                    num_neurons / (i + 2),
                    kernel_initializer="glorot_normal",
                    activation="relu",
                )
            )

        # Predict the target variable Y in % [0,100]
        model.add(layers.Dense(1, activation="sigmoid"))
        self.generator = model

    def make_discriminator(
        self,
        num_neurons: int,
        num_layers: int,
        input_shape: (int, None),
        num_sensitive_groups: int,
    ):
        """Create Discriminator Network"""
        model = tf.keras.Sequential()

        model.add(
            layers.Dense(
                num_neurons,
                input_shape=input_shape,
                kernel_initializer="glorot_normal",
                activation="relu",
            )
        )

        for i in range(num_layers - 1):
            model.add(
                layers.Dense(
                    num_neurons / (i + 2),
                    kernel_initializer="glorot_normal",
                    activation="relu",
                )
            )

        # Predict the sensitive group in % [0,100]
        model.add(layers.Dense(num_sensitive_groups, activation="sigmoid"))
        self.discriminator = model

    def _discriminator_loss(self, true_sensitive_values, pred_sensitive_value):
        no_clue_loss = self.cross_entropy(
            true_sensitive_values, pred_sensitive_value
        )  # Wie oft hat disc wahres S nicht erkannt
        return no_clue_loss

    def _generator_loss(
        self,
        true_target_value,
        pred_target_value,
        true_sensitive_values,
        pred_sensitive_value,
    ):
        precision_loss = self.cross_entropy(
            true_target_value, pred_target_value
        )  # Wie ungenau war Vorhersage für Y
        no_clue_loss = self.cross_entropy(
            true_sensitive_values, pred_sensitive_value
        )  # Wie oft hat disc wahres S nicht erkannt
        return precision_loss - self.fairness_multiplier * no_clue_loss

    def train_step(self, observation):
        feature_vector, true_sensitive_values, true_target_value = prepare_model_input(
            self, observation
        )

        # Watch Gradients for this context
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            # Calculate target Y with Generator
            generator_input = prepare_generator_input(
                feature_vector, true_sensitive_values
            )
            pred_target_value = self.generator(generator_input, training=True)

            # Calculate senstive values S with Discriminator
            discriminator_input = prepare_discriminator_input(
                feature_vector,
                pred_target_value,
                true_target_value,
                self.disc_observation_based,
            )
            pred_sensitive_value = self.discriminator(
                discriminator_input, training=True
            )

            # Reformat true target/senstive values
            true_target_value = tf.reshape(true_target_value, pred_target_value.shape)
            true_sensitive_values = tf.reshape(
                true_sensitive_values, pred_sensitive_value.shape
            )

            # Calculate Generator / Discriminator loss
            gen_loss = self._generator_loss(
                true_target_value,
                pred_target_value,
                true_sensitive_values,
                pred_sensitive_value,
            )
            disc_loss = self._discriminator_loss(
                true_sensitive_values, pred_sensitive_value
            )

        # Calclate gradients of Generator / Discriminator Loss over trainable variables
        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        # print(f"Generator Trainable Variables: {self.generator.trainable_variables}")
        # print(f"Generator Gradients: {gradients_of_generator}")

        # Apply Generator / Discriminator gradients to trainable variables
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )

        return gen_loss, disc_loss

    def train(self, dataset, epochs, batch_size):
        print(f"Start Training - Total of {epochs} Epochs:\n")

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

        for epoch in range(epochs):
            print("-------------------------------------------")
            print(f"Beginning of Epoch: {epoch+1}\n")
            start = time.time()

            for ind, observations in dataset.groupby(
                np.arange(len(dataset)) // batch_size
            ):
                gen_loss, disc_loss = self.train_step(observations)
                gen_loss_in_epoch.append(gen_loss)
                disc_loss_in_epoch.append(disc_loss)

            gen_loss_series.append(np.mean(gen_loss_in_epoch))
            disc_loss_series.append(np.mean(disc_loss_in_epoch))
            print(f"Generator Loss: {np.mean(gen_loss_in_epoch)}")
            print(f"Discriminator Loss: {np.mean(disc_loss_in_epoch)}")

            # Save the model every 10 epochs
            if (epoch + 1) % 100 == 0:
                save_model(self, np.mean(gen_loss_in_epoch))

            # Save fairness metrics every few epochs
            if epoch in self.save_fairness_in:
                fairness_series = save_fairness_metrics(
                    self, dataset.copy(), fairness_series, epoch
                )

            print(f"Time for epoch {epoch + 1} is {time.time()-start} sec\n")

        return gen_loss_series, disc_loss_series, fairness_series

    def generate_prediction(self, validation_df):
        validation_df = np.asmatrix(validation_df)
        return self.generator(validation_df, training=False).numpy()
