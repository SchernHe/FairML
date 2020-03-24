"""Deep Wasserstein Adversarial Network to mitigate bias
"""

import tensorflow as tf
from tensorflow.keras import layers
import time
import numpy as np
from fairml.metrics.consistency import calculate_consistency, fit_nearest_neighbors


def _create_batches(dataset, batch_size):
    """Helper Function to create batches. Each batch is a list of two samples"""

    idx = (len(dataset) // batch_size) * batch_size
    dataset = dataset.values
    np.random.shuffle(dataset)
    subset = dataset[0:idx, :]

    num_of_batches = len(dataset) // batch_size

    batches = [
        _retrieve_samples(batch, batch_size)
        for batch in np.array_split(subset, num_of_batches)
    ]

    return batches


def _retrieve_samples(batch, batch_size):
    """Helper function to create samples from each batch"""
    sample_size = int(batch_size / 2)
    sample_1 = batch[0:sample_size, :]
    sample_2 = batch[sample_size:, :]
    return (sample_1, sample_2)


def generate_prediction(model, validation_df):
    """Helper function to generate predictions, given a model and a dataframe"""
    validation_df = np.asmatrix(validation_df)
    return model.generator(validation_df, training=False).numpy()


class Individual_FairWAN:
    def __init__(
        self,
        G_optimizer,
        C_optimizer,
        x_idx,
        s_idx,
        y_idx,
        i_idx,
        mode_critic,
        mode_generator,
    ):
        tf.keras.backend.set_floatx("float64")
        self.G_optimizer = G_optimizer
        self.C_optimizer = C_optimizer
        self.x_idx = x_idx
        self.s_idx = s_idx
        self.y_idx = y_idx
        self.i_idx = i_idx
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mode_critic = mode_critic
        self.mode_generator = mode_generator

    def make_generator(self, num_neurons: int, input_shape: (int, None)):
        """Create Generator Network

        Parameters
        ----------
        num_neurons : int
            Description
        input_shape : int, None
            Description
        """

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
                int(num_neurons), kernel_initializer="glorot_normal", activation="relu",
            )
        )

        # Predict the target variable Y in % [0,100]
        generator.add(layers.Dense(1, activation="sigmoid"))
        self.generator = generator

    def make_critic(
        self, num_neurons: int, input_shape: (int, None),
    ):
        """Create Critic Network

        Parameters
        ----------
        num_neurons : int
            Description
        input_shape : int, None
            Description
        """
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
                num_neurons,
                input_shape=input_shape,
                kernel_initializer="glorot_normal",
                activation="relu",
            )
        )

        critic.add(layers.Dense(1))
        self.critic = critic

    @tf.function
    def calculate_critic_loss(self, C_real, C_fake):
        """Calculate critic  loss

        Parameters
        ----------
        C_fake : tf.tensor
            Critic output real
        C_fake : tf.tensor
            Critic output fake

        Returns
        -------
        critic_loss
        """

        w_distance = tf.reduce_mean(C_fake) - tf.reduce_mean(C_real)
        l2_penalty = self.mode_critic.get("l2_penalty") * tf.add_n(
            [
                tf.nn.l2_loss(v)
                for v in self.critic.trainable_variables
                if "bias" not in v.name
            ]
        )

        return w_distance + l2_penalty

    @tf.function
    def calculate_generator_loss(self, C_fake, Y_HAT, Y, batch_size):
        """Calculate generator loss

        Parameters
        ----------
        C_fake : tf.tensor
            Critic output fake
        Y_HAT : tf.tensor
            Generator predictions of samle one
        Y : tf.tensor
            True y values of sample one
        batch_size : int

        Returns
        -------
        generator_loss : tf.float64
        """

        w_distance = -tf.reduce_mean(C_fake)
        l2_penalty = self.mode_generator.get("l2_penalty") * tf.add_n(
            [
                tf.nn.l2_loss(v)
                for v in self.generator.trainable_variables
                if "bias" not in v.name
            ]
        )

        # Activate Cross-Entropy with Lambda Parameter
        if self.mode_generator.get("lambda"):
            generator_loss = self.cross_entropy(Y, Y_HAT) + self.mode_generator.get("lambda") * w_distance + l2_penalty
        
        generator_loss = w_distance + l2_penalty

        return generator_loss

    @tf.function
    def train_step(
        self,
        G_input,
        C_input_real,
        Y,
        X,
        batch_size,
        train_generator=False,
        train_critic=True,
    ):
        """Summary

        Parameters
        ----------
        G_input : np.matrix
        C_input_real : np.matrix
        Y : np.matrix
            Y Values of Sample One
        X : np.matrix
            X Values of Sample One
        batch_size : int
        train_generator : bool, optional
            Flag whether to train generator
        train_critic : bool, optional
            FLag whether to train critic

        Returns
        -------
        List
            Calculated Generator and Critic Loss
        """

        with tf.GradientTape() as G_tape, tf.GradientTape() as C_tape:

            G_output = self.generator(G_input, training=train_generator)

            C_input_fake = tf.concat([X, G_output], axis=1)

            # Calculate C(X) and C(G(x))
            C_fake = self.critic(C_input_fake, training=train_critic)
            C_real = self.critic(C_input_real, training=train_critic)

            # Calculate Generator and Critic loss_in_epoch
            C_loss = self.calculate_critic_loss(C_real, C_fake)
            G_loss = self.calculate_generator_loss(C_fake, G_output, Y, batch_size)

            C_loss += 0.001 * self._add_gradient_penalty(
                C_input_real, C_input_fake, batch_size
            )

            # Train Generator
            G_gradients = G_tape.gradient(G_loss, self.generator.trainable_variables)
            self.G_optimizer.apply_gradients(
                zip(G_gradients, self.generator.trainable_variables)
            )

            # Train Critic
            C_gradients = C_tape.gradient(C_loss, self.critic.trainable_variables)

            self.C_optimizer.apply_gradients(
                zip(C_gradients, self.critic.trainable_variables)
            )

            return [G_loss, C_loss]

    def train(
        self,
        dataset,
        epochs_total,
        epochs_critic,
        batch_size,
        informative_variables,
        save_consistency_score=False,
        num_knn=10,
    ):
        """Training Procedure

        Parameters
        ----------
        dataset : pd.DataFrame
        epochs_total : int
        epochs_critic : int
        batch_size : int
        informative_variables : list
            Colnames of "informative" features
        save_consistency_score : bool, optional
            Flag whether to save the consistency score every 10 epochs
        num_knn : int
            Number of KNN for consistency score

        """
        print(f"Start Training - Total of {epochs_total} Epochs:\n")

        # Initialize Placeholder
        loss_in_epoch = np.array([0.0, 0.0])
        G_loss_in_epoch_series, C_loss_in_epoch_series, Consistency_score_series = (
            [],
            [],
            [],
        )

        neigh = fit_nearest_neighbors(dataset[informative_variables], num_knn)

        for epoch in range(epochs_total):
            start = time.time()

            print("-------------------------------------------")
            print(f"Beginning of Epoch: {epoch+1}\n")

            for _ in range(epochs_critic):
                # Train only Critic
                for sample in _create_batches(dataset, batch_size):
                    G_input, C_input_real, Y, X = self._prepare_inputs(sample)
                    self.train_step(
                        G_input, C_input_real, Y, X, batch_size, False, True
                    )

            for sample in _create_batches(dataset, batch_size):
                # Train Generator and Critic
                G_input, C_input_real, Y, X = self._prepare_inputs(sample)
                sample_loss_in_epoch = (1 / batch_size) * np.array(
                    [
                        loss.numpy()
                        for loss in self.train_step(
                            G_input, C_input_real, Y, X, batch_size, True, True
                        )
                    ]
                )

                loss_in_epoch += sample_loss_in_epoch

            # Saving results
            G_loss_in_epoch_series.append(loss_in_epoch[0])
            C_loss_in_epoch_series.append(loss_in_epoch[1])

            if save_consistency_score and ((epoch == 2) or ((epoch % 10) == 0)):
                Consistency_score_series = self._save_consistency_score(
                    dataset.copy(),
                    Consistency_score_series,
                    informative_variables,
                    neigh,
                )

            # Output results
            print(f"Generator Loss: {loss_in_epoch[0]}")
            print(f"Critic Loss (W-Distance): {loss_in_epoch[1]}")
            print(f"Time for epoch {epoch + 1} is {time.time()-start} sec\n")

        return G_loss_in_epoch_series, C_loss_in_epoch_series, Consistency_score_series

    def _prepare_inputs(self, samples):
        """Helper Function: Prepare input vectors for model

        Parameters
        ----------
        samples : np.matrix

        Returns
        -------
        G_input: np.matrix
            Generator Input of size (Batch, Columns)
        C_input_real: np.matrix
            Critic Input Sample Two of size (Batch, Columns)
        Y: np.matrix
            True Y values of Sample One (Batch, 1)
        X: np.matrix
            X Values of Sample One (Batch, Columns)
        """

        G_input = samples[0][:, self.x_idx + self.s_idx]

        C_input_real = tf.concat(
            (samples[1][:, self.i_idx], samples[1][:, self.y_idx]), axis=1
        )
        Y = samples[0][:, self.y_idx]
        X = samples[0][:, self.i_idx]

        return G_input, C_input_real, Y, X

    @tf.function
    def _add_gradient_penalty(self, C_input_real, C_input_fake, batch_size):
        """Helper Function: Add gradient penalty to enforce Lipschitz continuity

        Parameters
        ----------
        C_input_real : np.matrix
            Critic Input Real (Sample Two)
        C_input_fake : tf.Tensor
            Critic Input Fake (Sample 2 X with generator predictions)
        batch_size : int

        Returns
        -------
        tf.tensor of type tf.float64
            Gradient penalty term
        """
        alpha = tf.random.uniform(
            shape=[int(batch_size / 2), 1], minval=0.0, maxval=1.0, dtype=tf.float64
        )

        interpolates = alpha * C_input_real + ((1 - alpha) * C_input_fake)
        disc_interpolates = self.critic(interpolates)
        gradients = tf.gradients(disc_interpolates, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
        return gradient_penalty

    def _save_consistency_score(
        self, df, Consistency_score_series, informative_variables, neigh
    ):
        """Helper Function: Save the consistency score to Series

        Parameters
        ----------
        df : pd.DataFrame
        Consistency_score_series : list
        informative_variables : list
            Colnames of "informative" features
        neigh : Fitted KNN model

        """

        print(f"---- Calculate Consitency Value!")

        df["Y_SCORE"] = generate_prediction(self, df.values[:, self.x_idx + self.s_idx])
        df["Y"] = df["Y_SCORE"].apply(lambda row: 1 if row > 0.5 else 0)

        consistency_in_epoch = calculate_consistency(
            df[informative_variables + ["Y"]].copy(), "Y", neigh
        )
        Consistency_score_series.append(consistency_in_epoch)

        print(f"---- Consitency Score: {consistency_in_epoch}")

        return Consistency_score_series
