"""Deep Wasserstein Adversarial Network to mitigate bias
"""

import tensorflow as tf
from tensorflow.keras import layers
import time
import numpy as np


class Individual_FairWAN:
    """Fair-WGAN Model"""

    def __init__(self, G_optimizer, C_optimizer):
        """Summary
        
        Parameters
        ----------
        G_optimizer : tf.Optimizer
            Generator Optimizer
        C_optimizer : tf.Optimizer
            Critic Optimizer
        """
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
        lambda_wasserstein,
        lambda_regularization,
        lambda_gradient_penalty,
        use_gradient_penalty,
    ):
        """Train Fair-WGAN model
        
        Parameters
        ----------
        dataset : pd.DataFrame
            Dataframe with samples
        batches : list
            List of samples provided by KNNSampler
        sampler : KNNSampler
            Sampler engine
        epochs_total : int
            Number of total epochs
        epochs_critic : int
            Number of critic iterations for each generator iteration
        batch_size : int
            Number of samples in each batch
        lambda_wasserstein : float
            Mutiplier for wasserstein term
        lambda_regularization : float
            Mutiplier for regularization term
        lambda_gradient_penalty: float
            Multiplier for gradient penalty term
        use_gradient_penalty : boold
            Boolean flag indicating to use gradient penalty or gradient clipping

        Returns
        -------
            Series of losses in each epoch [BCE_loss, W_loss, C_loss]

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
                        lambda_wasserstein=tf.constant(
                            lambda_wasserstein, dtype=tf.float64
                        ),
                        lambda_regularization=tf.constant(
                            lambda_regularization, dtype=tf.float64
                        ),
                       lambda_gradient_penalty=tf.constant(
                            lambda_gradient_penalty, dtype=tf.float64
                        ),
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
                            lambda_wasserstein=tf.constant(
                                lambda_wasserstein, dtype=tf.float64
                            ),
                            lambda_regularization=tf.constant(
                                lambda_regularization, dtype=tf.float64
                            ),
                            lambda_gradient_penalty=tf.constant(
                                lambda_gradient_penalty, dtype=tf.float64
                            ),
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
    C_input_gp,
    lambda_wasserstein,
    lambda_regularization,
    lambda_gradient_penalty,
    train_generator,
    train_critic,
    use_gradient_penalty,
):
    """Training step of Fair-WGAN model
    
    Parameters
    ----------
    generator : tf.Sequential
        Generator neural network
    critic : tf.Sequential
        Critic neural network
    G_optimizer : tf.Optimizer
        Generator Optimizer
    C_optimizer : tf.Optimizer
        Critic Optimizer
    G_input : tf.Tensor
        Description
    C_input_real : tf.Tensor
        Description
    Y_target : tf.Tensor
        Description
    C_input_gp : tf.Tensor
        Description
    lambda_wasserstein : tf.tensor(dtype=tf.Float64)
        Mutiplier for wasserstein term
    lambda_regularization : tf.tensor(dtype=tf.Float64)
        Mutiplier for regularization term
    lambda_gradient_penalty: tf.tensor(dtype=tf.Float64)
        Multiplier for gradient penalty term
    train_generator : bool
        Boolean flag indicating to train generator
    train_critic : bool
        Boolean flag indicating to train critic
    use_gradient_penalty : boold
        Boolean flag indicating to use gradient penalty or gradient clipping
    
    Returns
    -------
        Loss in iteration [BCE_loss, W_loss, C_loss]

    """

    with tf.GradientTape() as G_tape, tf.GradientTape() as C_tape:
        G_output = generator(G_input, training=train_generator)
        C_input_fake = tf.transpose(G_output)

        # Calculate C(X) and C(G(x))
        C_fake = critic(C_input_fake, training=train_critic)
        C_real = critic(C_input_real, training=train_critic)

        # Calculate Generator and Critic loss_in_epoch
        C_loss = calculate_critic_loss(
            critic,
            C_real=C_real,
            C_fake=C_fake,
            lambda_regularization=lambda_regularization,
        )

        generator_loss = calculate_generator_loss(
            generator,
            C_fake=C_fake,
            Y_HAT=G_output,
            Y=Y_target,
            lambda_wasserstein=lambda_wasserstein,
            lambda_regularization=lambda_regularization,
        )

        G_loss = generator_loss[0]
        cross_entropy_loss = generator_loss[1]
        wasserstein_loss = generator_loss[2]

        if use_gradient_penalty:
            C_loss += lambda_gradient_penalty * add_gradient_penalty(critic, C_input_gp, C_input_fake)

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


def calculate_generator_loss(
    generator, C_fake, Y_HAT, Y, lambda_wasserstein, lambda_regularization
):
    """Calculate generator loss term
     
    Parameters
    ----------
    generator : tf.Sequential
        Generator neural network
    C_fake : tf.Tensor
        Critic output of Generator(X)
    Y_HAT : tf.Tensor
        Generator predictions of target labels
    Y : tf.Tensor
        True labels
    lambda_wasserstein : tf.tensor(dtype=tf.Float64)
        Mutiplier for wasserstein term
    lambda_regularization : tf.tensor(dtype=tf.Float64)
        Mutiplier for regularization term

    Returns
    -------
    tf.tensor(dtype=tf.Float64)
        Generator Loss = Binary Cross Entropy + Wasserstein-1 + Regularization

    """

    regularization = tf.add_n(
        [
            tf.nn.l2_loss(v)
            for v in generator.trainable_variables
            if "bias" not in v.name
        ]
    )

    w_distance = -tf.reduce_mean(C_fake)

    cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()(Y, Y_HAT)
    wasserstein_loss = lambda_wasserstein * w_distance
    regularization_loss = lambda_regularization * regularization

    generator_loss = cross_entropy_loss + wasserstein_loss + regularization_loss

    return [generator_loss, cross_entropy_loss, wasserstein_loss]


def calculate_critic_loss(critic, C_real, C_fake, lambda_regularization):
    """Calculate critic loss term
    
    Parameters
    ----------
    critic : tf.Sequential
        Critic neural network
    C_real : tf.Tensor
        Critic output of target values of similar samples provided by the Sampler
    C_fake : tf.Tensor
        Critic output of Generator(X)
    lambda_regularization : tf.tensor(dtype=tf.Float64)
        Mutiplier for regularization term
    
    Returns
    -------
    tf.tensor(dtype=tf.Float64)
        Critic Loss = Wasserstein-1 + Regularization

    """

    regularization = tf.add_n(
        [tf.nn.l2_loss(v) for v in critic.trainable_variables if "bias" not in v.name]
    )

    w_distance = tf.reduce_mean(C_fake) - tf.reduce_mean(C_real)
    return w_distance + lambda_regularization * regularization


def add_gradient_penalty(critic, C_input_gp, C_input_fake):
    """Helper Function: Add gradient penalty to enforce Lipschitz continuity
        Interpolates = Real - alpha * ( Fake - Real )
    
    Parameters
    ----------
    critic : tf.Sequential
        Critic neural network
    C_input_gp : np.matrix
        Critic input for gradient penalty. Mean values of all similar samples
        provided by the Sampler.
    C_input_fake : tf.Tensor
        Critic input Generator(X)
    
    Returns
    -------
    tf.tensor(dtype=tf.Float64)
        Gradient Penalty

    """

    alpha = tf.random.uniform(
        shape=[1, int(C_input_fake.shape[1])], minval=0.0, maxval=1.0, dtype=tf.float64
    )

    interpolates = C_input_gp + alpha * (C_input_fake - C_input_gp)

    disc_interpolates = critic(interpolates)
    gradients = tf.gradients(disc_interpolates, [interpolates])[0]

    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))

    return tf.reduce_mean((slopes - 1) ** 2)


def _print_and_append_loss(
    BCE_loss_series, W_loss_series, C_loss_series, loss_in_epoch
):
    """Helper function to print and save loss values in each epoch

    Parameters
    ----------
    BCE_loss_series : list
        Binary Cross-Entropy loss values
    W_loss_series : list
        Wasserstein-1 loss values in generator
    C_loss_series : list
        Wasserstein-1 loss values in critic
    loss_in_epoch : list
        Losses in the epoch of the form [BCE_Loss, W_Loss, C_Loss]
    
    Returns
    -------
    List of losses [BCE_loss, W_loss, C_loss]

    """

    BCE_loss_series.append(loss_in_epoch[0])
    W_loss_series.append(loss_in_epoch[1])
    C_loss_series.append(loss_in_epoch[2])

    # Output results
    print(f"Generator - Binary-Cross Entropy Loss: {loss_in_epoch[0]}")
    print(f"Generator - Wasserstein Loss: {loss_in_epoch[1]}")
    print(f"Critic - Wasserstein Distance: {loss_in_epoch[2]}")

    return BCE_loss_series, W_loss_series, C_loss_series
