"""Deep Wasserstein Adversarial Network to mitigate bias
"""

import tensorflow as tf
from tensorflow.keras import layers
import time
import numpy as np
from fairml.metrics.consistency import calculate_consistency


def generate_prediction(model, df):
    """Helper function to generate predictions, given a model and a dataframe"""
    df = np.asmatrix(df)
    return model(df, training=False).numpy()


class Individual_FairWAN:
    def __init__(
        self, G_optimizer, C_optimizer, mode_critic, mode_generator,
    ):
        tf.keras.backend.set_floatx("float64")
        self.G_optimizer = G_optimizer
        self.C_optimizer = C_optimizer
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
        generator._name = "Generator"

        generator.add(
            layers.Dense(
                num_neurons,
                input_shape=input_shape,
                kernel_initializer="glorot_normal",
                activation="relu",
            )
        )

        generator.add(
            layers.Dense(32, kernel_initializer="glorot_normal", activation="relu",)
        )

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
        critic._name = "Critic"

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

    def train(
        self,
        dataset,
        sampled_batches,
        sampler,
        epochs_total,
        epochs_critic,
        batch_size,
        informative_variables,
        num_knn=10,
    ):
        """Training Procedure

        Parameters
        ----------
        dataset : pd.DataFrame
        sampler : fairml.models.Sampler
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
        print(f"Got total number of {len(sampled_batches)} batches!")
        # Initialize Placeholder
        G_loss_in_epoch_series, C_loss_in_epoch_series, Consistency_score_series = (
            [],
            [],
            [],
        )
        sample_count = 0

        for epoch in range(epochs_total):
            start = time.time()
            print("-------------------------------------------")
            print(f"Beginning of Epoch: {epoch+1}\n")

            # Take new list of batches
            batches = sampled_batches[sample_count]
            sample_count += 1
            
            if sample_count == len(sampled_batches):
                sample_count = 0

            if epoch == 0:
                for batch in batches:
                    input_data = sampler._prepare_inputs(dataset, batch)

                    train_step(
                        self.generator,
                        self.critic,
                        self.G_optimizer,
                        self.C_optimizer,
                        *input_data,
                        tf.constant(True),
                        tf.constant(True),
                    )

            for _ in range(epochs_critic):
                # Train only Critic
                for batch in batches:
                    input_data = sampler._prepare_inputs(dataset, batch)

                    train_step(
                        self.generator,
                        self.critic,
                        self.G_optimizer,
                        self.C_optimizer,
                        *input_data,
                        tf.constant(False),
                        tf.constant(True),
                    )

            loss_in_epoch = np.array([0.0, 0.0])

            for batch in batches:
                input_data = sampler._prepare_inputs(dataset, batch)

                batch_loss = np.array(
                    [
                        loss.numpy()
                        for loss in train_step(
                            self.generator,
                            self.critic,
                            self.G_optimizer,
                            self.C_optimizer,
                            *input_data,
                            tf.constant(True),
                            tf.constant(True),
                        )
                    ]
                )

                loss_in_epoch += batch_loss

            # Saving results
            G_loss_in_epoch_series.append(loss_in_epoch[0])
            C_loss_in_epoch_series.append(loss_in_epoch[1])

            # Output results
            print(f"Generator Loss: {loss_in_epoch[0]}")
            print(f"Critic Loss (W-Distance): {loss_in_epoch[1]}")
            print(f"Time for epoch {epoch + 1} is {time.time()-start} sec\n")

        return G_loss_in_epoch_series, C_loss_in_epoch_series, Consistency_score_series


@tf.function(experimental_relax_shapes=True)
def train_step(
    generator,
    critic,
    G_optimizer,
    C_optimizer,
    G_input,
    C_input_real,
    Y_target,
    Mean_C_input_real,
    train_generator,
    train_critic,
):
    """Summary

    Parameters
    ----------
    G_input : np.matrix
    C_input_real : np.matrix
    Y_target : np.matrix
        Y Values of Sample One
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

        G_output = generator(G_input, training=train_generator)

        C_input_fake = G_output

        # Calculate C(X) and C(G(x))
        C_fake = critic(C_input_fake, training=train_critic)
        C_real = critic(C_input_real, training=train_critic)

        # Calculate Generator and Critic loss_in_epoch
        C_loss = calculate_critic_loss(critic, C_real, C_fake)
        G_loss = calculate_generator_loss(
            generator, C_fake=C_fake, Y_HAT=G_output, Y=Y_target
        )

        gradient_penalty = tf.constant(10., dtype=tf.float64) * add_gradient_penalty(
            critic, Mean_C_input_real, C_input_fake
        )
        #print("Critic loss %f %f" % (C_loss ,gradient_penalty))
        C_loss += gradient_penalty

        # Train Generator
        G_gradients = G_tape.gradient(G_loss, generator.trainable_variables)
        G_optimizer.apply_gradients(zip(G_gradients, generator.trainable_variables))

        # Train Critic
        C_gradients = C_tape.gradient(C_loss, critic.trainable_variables)

        C_optimizer.apply_gradients(zip(C_gradients, critic.trainable_variables))

        # Gradient Clipping
        # if required
        # clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.critic.trainable_variables]

        return [G_loss, C_loss]


@tf.function(experimental_relax_shapes=True)
def calculate_generator_loss(generator, C_fake, Y_HAT, Y):
    """Calculate generator loss

    Parameters
    ----------
    C_fake : tf.tensor
        Critic output fake
    Y_HAT : tf.tensor
        Generator predictions of samle one
    Y : tf.tensor
        True y values of sample one

    Returns
    -------
    generator_loss : tf.float64
    """
    
    lambda_wasserstein = tf.constant(100, dtype=tf.float64)

    w_distance = -tf.reduce_mean(C_fake)

    lambda_l2 = tf.constant(0.01, dtype=tf.float64)
    l2_penalty = tf.add_n(
        [
            tf.nn.l2_loss(v)
            for v in generator.trainable_variables
            if "bias" not in v.name
        ]
    )

    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    cross_entropy_loss = cross_entropy(Y, Y_HAT)
    wasserstein_loss = lambda_wasserstein * w_distance
    regularization_loss = lambda_l2 * l2_penalty

    #        print("Loss terms %f , %f, %f" % (cross_entropy_loss,wasserstein_loss,regularization_loss))

    return cross_entropy_loss + wasserstein_loss + regularization_loss


@tf.function(experimental_relax_shapes=True)
def calculate_critic_loss(critic, C_real, C_fake):
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

    lambda_l2 = tf.constant(0.01, dtype=tf.float64)

    l2_penalty = lambda_l2 * tf.add_n(
        [tf.nn.l2_loss(v) for v in critic.trainable_variables if "bias" not in v.name]
    )

    return tf.math.add(w_distance, l2_penalty)


@tf.function(experimental_relax_shapes=True)
def add_gradient_penalty(critic, Mean_C_input_real, C_input_fake):
    """Helper Function: Add gradient penalty to enforce Lipschitz continuity

        Interpolates = Real - alpha * ( Fake - Real )


    Parameters
    ----------
    C_input_real : np.matrix
        Critic Input Real (Sample Two)
    C_input_fake : tf.Tensor
        Critic Input Fake (Sample 2 X with generator predictions)

    Returns
    -------
    tf.tensor of type tf.float64
        Gradient penalty term
    """
    alpha = tf.random.uniform(
        shape=[1, int(C_input_fake.shape[1])], minval=0.0, maxval=1.0, dtype=tf.float64
    )

    interpolates = Mean_C_input_real + alpha * (C_input_fake - Mean_C_input_real)

    disc_interpolates = critic(interpolates)
    gradients = tf.gradients(disc_interpolates, [interpolates])[0]

    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))

    gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)

    return gradient_penalty
