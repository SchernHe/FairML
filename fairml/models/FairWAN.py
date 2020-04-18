"""Deep Wasserstein Adversarial Network to mitigate bias
"""

import tensorflow as tf
from tensorflow.keras import layers
import time
import numpy as np


class Individual_FairWAN:
    def __init__(self, G_optimizer, C_optimizer):
        tf.keras.backend.set_floatx("float64")
        self.G_optimizer = G_optimizer
        self.C_optimizer = C_optimizer

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
            layers.Dense(
                num_neurons, kernel_initializer="glorot_normal", activation="relu"
            )
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
                num_neurons, kernel_initializer="glorot_normal", activation="relu",
            )
        )

        critic.add(layers.Dense(1))
        self.critic = critic

    def train(
        self,
        dataset,
        batches,
        sampler,
        epochs_total,
        epochs_critic,
        batch_size,
        LAMBDA,
        use_gradient_penalty,
    ):
        """Training Procedure

        Parameters
        ----------
        dataset : pd.DataFrame
        sampler : fairml.models.Sampler
        epochs_total : int
        epochs_critic : int
        batch_size : int

        """
        print(f"Start Training - Total of {epochs_total} Epochs:\n")
        print(f"Got total number of {len(batches)} batches!")

        # Initialize Placeholder
        BCE_loss_series, W_loss_series, C_loss_series = (
            [],
            [],
            [],
        )

        for epoch in range(epochs_total):
            start = time.time()
            print("-------------------------------------------")
            print(f"Beginning of Epoch: {epoch+1}\n")

            loss_in_epoch = np.array([0.0, 0.0, 0.0])

            if epoch > int(epochs_total * 0.75):
                print("Train Generator Only")
                for batch in batches:
                    input_data = sampler._prepare_inputs(dataset, batch)

                    batch_loss = np.array(
                        [
                            loss.numpy()
                            for loss in _train(
                                self.generator,
                                self.critic,
                                self.G_optimizer,
                                self.C_optimizer,
                                *input_data,
                                LAMBDA=tf.constant(LAMBDA, dtype=tf.float64),
                                train_generator=True,
                                train_critic=False,
                                use_gradient_penalty=use_gradient_penalty,
                            )
                        ]
                    )

                    loss_in_epoch += (1 / len(batches)) * batch_loss

                BCE_loss_series, W_loss_series, C_loss_series = _print_and_append_loss(
                    BCE_loss_series, W_loss_series, C_loss_series, loss_in_epoch
                )
                continue

            # Train Critic Only
            for _ in range(epochs_critic):
                # Train only Critic
                for batch in batches:
                    input_data = sampler._prepare_inputs(dataset, batch)

                    _train(
                        self.generator,
                        self.critic,
                        self.G_optimizer,
                        self.C_optimizer,
                        *input_data,
                        LAMBDA=tf.constant(LAMBDA, dtype=tf.float64),
                        train_generator=False,
                        train_critic=True,
                        use_gradient_penalty=use_gradient_penalty,
                    )

            # Train Both Networks
            for batch in batches:
                input_data = sampler._prepare_inputs(dataset, batch)

                batch_loss = np.array(
                    [
                        loss.numpy()
                        for loss in _train(
                            self.generator,
                            self.critic,
                            self.G_optimizer,
                            self.C_optimizer,
                            *input_data,
                            LAMBDA=tf.constant(LAMBDA, dtype=tf.float64),
                            train_generator=True,
                            train_critic=True,
                            use_gradient_penalty=use_gradient_penalty,
                        )
                    ]
                )

                loss_in_epoch += (1 / len(batches)) * batch_loss

            BCE_loss_series, W_loss_series, C_loss_series = _print_and_append_loss(
                BCE_loss_series, W_loss_series, C_loss_series, loss_in_epoch
            )

            print(f"Time for epoch {epoch + 1} is {time.time()-start} sec\n")

        return BCE_loss_series, W_loss_series, C_loss_series


@tf.function(experimental_relax_shapes=True)
def _train(
    generator,
    critic,
    G_optimizer,
    C_optimizer,
    G_input,
    C_input_real,
    Y_target,
    Mean_C_input_real,
    LAMBDA,
    train_generator,
    train_critic,
    use_gradient_penalty,
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
        C_input_fake = tf.transpose(G_output)

        # Calculate C(X) and C(G(x))
        C_fake = critic(C_input_fake, training=train_critic)
        C_real = critic(C_input_real, training=train_critic)

        # Calculate Generator and Critic loss_in_epoch
        C_loss = calculate_critic_loss(critic, C_real, C_fake)

        generator_loss = calculate_generator_loss(
            generator,
            C_real=C_real,
            C_fake=C_fake,
            Y_HAT=G_output,
            Y=Y_target,
            LAMBDA=LAMBDA,
        )

        G_loss = generator_loss[0]
        cross_entropy_loss = generator_loss[1]
        wasserstein_loss = generator_loss[2]

        if use_gradient_penalty:
            C_loss += add_gradient_penalty(critic, Mean_C_input_real, C_input_fake)

        # Train Generator
        G_gradients = G_tape.gradient(G_loss, generator.trainable_variables)
        G_optimizer.apply_gradients(zip(G_gradients, generator.trainable_variables))

        # Train Critic
        C_gradients = C_tape.gradient(C_loss, critic.trainable_variables)

        C_optimizer.apply_gradients(zip(C_gradients, critic.trainable_variables))

        if not use_gradient_penalty:
            clip_D = [
                p.assign(tf.clip_by_value(p, -0.02, 0.02))
                for p in critic.trainable_variables
            ]

        return [cross_entropy_loss, wasserstein_loss, C_loss]


def calculate_generator_loss(generator, C_real, C_fake, Y_HAT, Y, LAMBDA):
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

    cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()(Y, Y_HAT)

    LAMBDA_L2 = tf.constant(0.01, dtype=tf.float64)
    regularization = tf.add_n(
        [
            tf.nn.l2_loss(v)
            for v in generator.trainable_variables
            if "bias" not in v.name
        ]
    )

    w_distance = -tf.reduce_mean(C_fake)
    wasserstein_loss = LAMBDA * w_distance

    generator_loss = cross_entropy_loss + wasserstein_loss + LAMBDA_L2 * regularization

    return [generator_loss, cross_entropy_loss, wasserstein_loss]


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
    LAMBDA_L2 = tf.constant(0.01, dtype=tf.float64)
    regularization = tf.add_n(
        [tf.nn.l2_loss(v) for v in critic.trainable_variables if "bias" not in v.name]
    )

    w_distance = tf.reduce_mean(C_fake) - tf.reduce_mean(C_real)
    return w_distance + LAMBDA_L2 * regularization


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

    gradient_penalty = tf.constant(10.0, dtype=tf.float64) * tf.reduce_mean(
        (slopes - 1) ** 2
    )

    return gradient_penalty


def _print_and_append_loss(
    BCE_loss_series, W_loss_series, C_loss_series, loss_in_epoch
):
    # Saving results
    BCE_loss_series.append(loss_in_epoch[0])
    W_loss_series.append(loss_in_epoch[1])
    C_loss_series.append(loss_in_epoch[2])

    # Output results
    print(f"Generator - Binary-Cross Entropy Loss: {loss_in_epoch[0]}")
    print(f"Generator - Wasserstein Loss: {loss_in_epoch[1]}")
    print(f"Critic - Wasserstein Distance: {loss_in_epoch[2]}")

    return BCE_loss_series, W_loss_series, C_loss_series
