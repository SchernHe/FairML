"""Deep Wasserstein Adversarial Network to mitigate bias
"""

import tensorflow as tf
from tensorflow.keras import layers
import time
import numpy as np
from fairml.metrics.consistency import calculate_consistency, fit_nearest_neighbors


def generate_prediction(model, df):
    """Helper function to generate predictions, given a model and a dataframe"""
    df = np.asmatrix(df)
    return model(df, training=False).numpy()


class Individual_FairWAN:
    def __init__(
        self,
        G_optimizer,
        C_optimizer,
        mode_critic,
        mode_generator,
    ):
        tf.keras.backend.set_floatx("float64")
        self.G_optimizer = G_optimizer
        self.C_optimizer = C_optimizer
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
                num_neurons, kernel_initializer="glorot_normal", activation="relu",
            )
        )

        generator.add(
            layers.Dense(
                num_neurons//2, kernel_initializer="glorot_normal", activation="relu",
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
                num_neurons//2,
                input_shape=input_shape,
                kernel_initializer="glorot_normal",
                activation="relu",
            )
        )


        critic.add(
            layers.Dense(
                num_neurons//2,
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
        lambda_l2 = tf.constant(self.mode_critic.get("l2_penalty",0.),dtype=tf.float64)

        l2_penalty = lambda_l2 * tf.add_n(
            [
                tf.nn.l2_loss(v)
                for v in self.critic.trainable_variables
                if "bias" not in v.name
            ]
        )
        critic_loss = tf.math.add(w_distance,l2_penalty)
        return critic_loss

    @tf.function
    def calculate_generator_loss(self, C_fake, Y_HAT, Y):
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
        lambda_wasserstein = tf.constant(self.mode_generator.get("lambda",1.),dtype=tf.float64)

        w_distance = -tf.reduce_mean(C_fake)

        lambda_l2 = tf.constant(self.mode_generator.get("l2_penalty",0.),dtype=tf.float64)
        l2_penalty = tf.add_n(
            [
                tf.nn.l2_loss(v)
                for v in self.generator.trainable_variables
                if "bias" not in v.name
            ]
        )

        generator_loss = lambda_wasserstein * w_distance + lambda_l2*l2_penalty

        # Activate Cross-Entropy with Lambda Parameter
        if self.mode_generator.get("lambda"):
            generator_loss += self.cross_entropy(Y, Y_HAT) 
            

        return generator_loss 

    @tf.function(experimental_relax_shapes=True)
    def train_step(
        self,
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

            G_output = self.generator(G_input, training=train_generator)
            C_input_fake = G_output
            # Calculate C(X) and C(G(x))
            C_fake = self.critic(C_input_fake, training=train_critic)
            C_real = self.critic(C_input_real, training=train_critic)

            # Calculate Generator and Critic loss_in_epoch
            C_loss = self.calculate_critic_loss(C_real, C_fake)
            G_loss = self.calculate_generator_loss(C_fake=C_fake, Y_HAT=G_output, Y=Y_target)

            gradient_penalty = 0.01 * add_gradient_penalty(
                self.critic, Mean_C_input_real, C_input_fake
            )
            C_loss += gradient_penalty

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
            
            #clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.critic.trainable_variables]
            
            return [G_loss, C_loss]


    def train(
        self,
        dataset,
        sampled_batches,
        sampler,
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
        loss_in_epoch = np.array([0.0, 0.0])
        G_loss_in_epoch_series, C_loss_in_epoch_series, Consistency_score_series = (
            [],
            [],
            [],
        )
        sample_count=0
        #neigh = fit_nearest_neighbors(dataset[informative_variables], num_knn)

        for epoch in range(epochs_total):
            start = time.time()
            print("-------------------------------------------")
            print(f"Beginning of Epoch: {epoch+1}\n")

            # Take new list of batches
   
            batches = sampled_batches[sample_count]
            sample_count +=1
            if sample_count ==len(sampled_batches):
                sample_count = 0

        
            for _ in range(epochs_critic):
                # Train only Critic
                for batch in batches:
                    G_input, C_input_real, Y_target, Mean_C_input_real = sampler._prepare_inputs(dataset,batch)
                    
                    self.train_step(
                        G_input, C_input_real, Y_target, Mean_C_input_real, tf.constant(False), tf.constant(True)
                    )


            for batch in batches:
                G_input, C_input_real, Y_target, Mean_C_input_real = sampler._prepare_inputs(dataset,batch)
                sample_loss_in_epoch = (1 / batch_size) * np.array(
                    [
                        loss.numpy()
                        for loss in self.train_step(
                            G_input, C_input_real, Y_target, Mean_C_input_real, tf.constant(True), tf.constant(True)
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

        df["Y_SCORE"] = generate_prediction(self.generator, df.values[:, self.x_idx + self.s_idx])
        df["Y"] = df["Y_SCORE"].apply(lambda row: 1 if row > 0.5 else 0)

        consistency_in_epoch = calculate_consistency(
            df[informative_variables + ["Y"]].copy(), "Y", neigh
        )
        Consistency_score_series.append(consistency_in_epoch)

        print(f"---- Consitency Score: {consistency_in_epoch}")

        return Consistency_score_series


@tf.function
def add_gradient_penalty(critic, Mean_C_input_real, C_input_fake):
    """Helper Function: Add gradient penalty to enforce Lipschitz continuity

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
        shape=[int(C_input_fake.shape[1]), 1], minval=0.0, maxval=1.0, dtype=tf.float64
    )
    differences = Mean_C_input_real - C_input_fake
    interpolates = Mean_C_input_real + alpha * differences
    disc_interpolates = critic(interpolates)
    gradients = tf.gradients(disc_interpolates, [interpolates])[0]
   
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
    
    gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
    
    return gradient_penalty