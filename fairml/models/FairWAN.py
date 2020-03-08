"""Deep Wasserstein Adversarial Network to mitigate bias
"""


import tensorflow as tf
from tensorflow.keras import layers
from fairml.models.utils import (
    save_model,
    save_fairness_metrics,
    prepare_model_input,
    prepare_generator_input,
    init_placeholders,
)
import time
import numpy as np
from sklearn.utils import shuffle
from scipy.stats import wasserstein_distance


class Individual_FairWAN:
    def __init__(
        self,
        G_optimizer,
        H_optimizer,
        C_optimizer,
        sensitive_variables,
        target_variable,
        checkpoint_dir,
        save_fairness_in,
        disc_observation_based,
    ):
        tf.keras.backend.set_floatx("float64")
        self.G_optimizer = G_optimizer
        self.H_optimizer = H_optimizer
        self.C_optimizer = C_optimizer
        self.sensitive_variables = sensitive_variables
        self.target_variable = target_variable
        self.checkpoint_dir = checkpoint_dir
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.save_fairness_in = save_fairness_in
        self.disc_observation_based = disc_observation_based

    def make_generator(self, num_neurons: int, input_shape: (int, None)):
        """Create Generator Network"""

        generator = tf.keras.Sequential()

        generator.add(
            layers.Dense(
                num_neurons,
                input_shape=input_shape,
                kernel_initializer="glorot_normal",
                activation="relu",
            )
        )

        generator.add(
            layers.Dense(
                int(num_neurons / 2),
                input_shape=input_shape,
                kernel_initializer="glorot_normal",
                activation="relu",
            )
        )

        # Predict the target variable Y in % [0,100]
        generator.add(layers.Dense(1, activation="sigmoid"))
        self.generator = generator

    def make_helper(self, num_neurons: int, input_shape: (int, None)):
        """Create helper Network"""
        helper = tf.keras.Sequential()

        helper.add(
            layers.Dense(
                num_neurons,
                input_shape=input_shape,
                kernel_initializer="glorot_normal",
                activation="relu",
            )
        )

        helper.add(
            layers.Dense(
                int(num_neurons / 2),
                input_shape=input_shape,
                kernel_initializer="glorot_normal",
                activation="relu",
            )
        )

        # Predict the sensitive group in % [0,100]
        helper.add(layers.Dense(1))
        self.helper = helper

    def make_critic(
        self, num_neurons: int, input_shape: (int, None),
    ):
        """Create helper Network"""
        critic = tf.keras.Sequential()

        critic.add(
            layers.Dense(
                num_neurons,
                input_shape=input_shape,
                kernel_initializer="glorot_normal",
                activation="relu",
            )
        )

        critic.add(
            layers.Dense(
                int(num_neurons / 2),
                input_shape=input_shape,
                kernel_initializer="glorot_normal",
                activation="relu",
            )
        )

        critic.add(layers.Dense(1))
        self.critic = critic

    def calculate_w_distance(self, real, fake):
        return tf.reduce_mean(real) - tf.reduce_mean(fake)

    def calculate_generator_loss(self, H_loss, C_fake):

        # Min. Wasserstein Distance Y and Y_HAT
        G_loss = -tf.reduce_mean(C_fake)

        # Scale W(Y,Y_HAT) by similarity of batch
        helper_tensor = tf.constant(1.,dtype=tf.float64)
        G_loss_scaled = tf.math.multiply(G_loss,tf.math.add(helper_tensor,tf.math.divide_no_nan(helper_tensor, tf.math.abs(H_loss))))

        return G_loss_scaled

    def train_step(self, sample_1, sample_2, batch_size, train_generator=False):

        # Watch Gradients for this context
        with tf.GradientTape() as G_tape, tf.GradientTape() as H_tape, tf.GradientTape() as C_tape:

            # Get real and fake output of helper
            H_real = self.helper(sample_1[0], training=True)
            H_fake = self.helper(sample_2[0], training=True)

            if train_generator:
                # Calculate target Y with Generator
                G_output = self.generator(sample_1[0], training=True)
            else:
                G_output = self.generator(sample_1[0], training=False)

            # Calculate Wasserstein Distance
            G_output = tf.reshape(G_output, (1, len(G_output)))  # 1,64
            C_input_real = tf.reshape(sample_1[1], (1, len(sample_1[1])))  # 1,64

            C_fake = self.critic(G_output, training=True)
            C_real = self.critic(C_input_real, training=True)

            # Calculate Generator / helper loss
            H_loss = self.calculate_w_distance(H_real, H_fake)
            C_loss = self.calculate_w_distance(C_real, C_fake)

            G_loss = self.calculate_generator_loss(H_loss, C_fake)

        H_gradients = H_tape.gradient(H_loss, self.helper.trainable_variables)
        C_gradients = C_tape.gradient(C_loss, self.critic.trainable_variables)

        H_gradients = [g * (-1) for g in H_gradients]
        C_gradients = [g * (-1) for g in C_gradients]

        self.H_optimizer.apply_gradients(
            zip(H_gradients, self.helper.trainable_variables)
        )

        self.C_optimizer.apply_gradients(
            zip(C_gradients, self.critic.trainable_variables)
        )

        if train_generator:
            G_gradients = G_tape.gradient(G_loss, self.generator.trainable_variables)
            self.G_optimizer.apply_gradients(
                zip(G_gradients, self.generator.trainable_variables)
            )

        # Clip helper weights to ensure Liptschitz Continuity
        for vars in self.helper.trainable_variables:
            vars = tf.clip_by_value(vars, -0.01, 0.01)

        # Clip critic weights to ensure Liptschitz Continuity
        for vars in self.critic.trainable_variables:
            vars = tf.clip_by_value(vars, -0.01, 0.01)

        return np.array([G_loss, H_loss, C_loss])

    def _create_batches(self, dataset, batch_size):
        tar_col = dataset.columns.get_loc(self.target_variable)
        idx = (len(dataset) // batch_size) * batch_size

        dataset = dataset.values
        np.random.shuffle(dataset)

        batches = [
            (
                batch[:, tar_col],
                np.concatenate((batch[:, :tar_col], batch[:, tar_col + 1 :]), axis=1),
            )
            for batch in np.array_split(dataset[:idx, :], len(dataset) // batch_size)
        ]
        # target , features

        return batches

    def train(self, dataset, epochs_total, epochs_w_only, batch_size):
        print(f"Start Training - Total of {epochs_total} Epochs:\n")
        print(f"First {epochs_w_only} Epochs only training critic and helper\n")

        loss = np.array([0.0, 0.0, 0.0])
        G_loss_series = []
        H_loss_series = []
        C_loss_series = []


        for epoch in range(epochs_total):

            print("-------------------------------------------")
            print(f"Beginning of Epoch: {epoch+1}\n")

            start = time.time()

            for target, features in self._create_batches(dataset, batch_size):
                index_to_divide = int(batch_size / 2)
                sample_1_target = target[0:index_to_divide]
                sample_1_features = features[0:index_to_divide]

                sample_2_target = target[index_to_divide:]
                sample_2_features = features[index_to_divide:]

                sample_1 = (sample_1_features, sample_1_target)
                sample_2 = (sample_2_features, sample_2_target)

                if epoch < epochs_w_only:
                    loss_in_epoch = (1 / batch_size) * self.train_step(
                        sample_1, sample_2, batch_size, False
                    )
                    loss += loss_in_epoch
                else:
                    # Train Generator
                    loss_in_epoch = (1 / batch_size) * self.train_step(
                        sample_1, sample_2, batch_size, True
                    )
                    loss += loss_in_epoch


            # Saving results
            G_loss_series.append(loss[0])
            H_loss_series.append(loss[1])
            C_loss_series.append(loss[2])

            # Save the model every 10 epochs
            # if (epoch + 1) % 100 == 0:
            #    save_model(self, np.mean(G_loss_in_epoch))

            # Output results
            print(f"Generator Loss: {loss[0]}")
            print(f"Helper Loss (W-Distance X): {loss[1]}")
            print(f"Critic Loss (W-Distance Y): {loss[2]}")
            print(f"Time for epoch {epoch + 1} is {time.time()-start} sec\n")

        return G_loss_series, H_loss_series, C_loss_series

    def generate_prediction(self, validation_df):
        validation_df = np.asmatrix(validation_df)
        return self.generator(validation_df, training=False).numpy()
