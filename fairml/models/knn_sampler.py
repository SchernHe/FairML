import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors


class KNNEngine:
    """ K-Neirest-Neighbors Engine"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.neigh = NearestNeighbors(metric="euclidean").fit(dataset)

    def __call__(self, observations, radius):
        neigh_dist, neigh_idx = self.neigh.radius_neighbors(observations, radius)
        return neigh_dist, neigh_idx


class KNNSampler:
    """Sampler Engine

    Attributes
    ----------
    dataset : pd.Dataframe
    KNNEngine : KNNEngine
    i_idx : list
        Indizes of informative columns
    s_idx : list
        Indizes of sensitve attributes
    x_idx : list
        Indizes of total features
    y_idx : list
        Indizes of Y target
    """

    def __init__(self, dataset, x_idx, s_idx, y_idx, i_idx):
        tf.keras.backend.set_floatx("float64")
        self.dataset = dataset.values
        self.KNNEngine = KNNEngine(self.dataset[:, i_idx])
        self.x_idx = x_idx
        self.s_idx = s_idx
        self.y_idx = y_idx
        self.i_idx = i_idx

    def __call__(self, batch_size=64, num_similar_samples=5, radius=0.0):
        """Execute sampling.

        Parameters
        ----------
        batch_size : int

        Returns
        -------
        list
            List of generated samples.
            Each element in the list corresponds to one decomposition of the dataset in n-batches.
            Each of these n-batches consists of 2 arrays, corresponding two the sample 1 and sample 2.
            The length of these samples are batch_size and k*batch_size

        """
        np.random.seed(30041994)
        total_idx = np.array(range(0, len(self.dataset) - 1))

        num_batches = len(total_idx) // batch_size

        batch_indizes = []

        for _ in range(0, num_batches):
            KNN_IDX = []

            observation_idx_knn = total_idx[
                (_ * batch_size) : (_ * batch_size + batch_size)
            ]
            slice_dataset = self.dataset[observation_idx_knn, :]

            neigh_dist, neigh_idx = self.KNNEngine(slice_dataset[:, self.i_idx], radius)

            for _, idx in enumerate(neigh_idx):

                if len(idx) == 0:
                    print("No IDX")
                    continue

                KNN_IDX.extend([np.random.choice(idx, num_similar_samples)])

            KNN_IDX = np.array(KNN_IDX)
            batch_indizes += [(observation_idx_knn, KNN_IDX)]

        return batch_indizes

    def _prepare_inputs(self, dataset, batch_idx: list):
        """Helper function to prepare model inputs.

        Parameters
        ----------
        dataset : pd.Dataframe
        batch_idx : list
            List, containing the indizes of the samples

            Sample 1: batch_idx[0]
                Containts batch-size elements of the dataframe.

            Sample 2: batch_idx[1]
                Containts batch-size * k elements of the dataframe.
                For each elemen in sample 1, there are kNN of this element in
                Sample 2.

        Returns
        -------
        list
            Containing the input vectors derived from the batch indizes

            G_input: Generator Input
            C_input_real: Real Y values of sample 1
            Y_target: Real Y values of sample 2
            C_input_gp: One NN of each element of sample 1
        """

        # Generator Input
        G_input = dataset.iloc[batch_idx[0], self.x_idx + self.s_idx].values
        Y_target = dataset.iloc[batch_idx[0], self.y_idx].values

        # Critic Input
        total_target_values = dataset.iloc[:, self.y_idx].values
        C_input_real = np.squeeze(total_target_values[batch_idx[1]])
        C_input_gp = C_input_real.mean(axis=1)

        # Convert to Tensor
        C_input_real = tf.transpose(
            tf.convert_to_tensor(C_input_real, dtype=tf.float64)
        )
        C_input_gp = tf.convert_to_tensor(C_input_gp, dtype=tf.float64)
        G_input = tf.convert_to_tensor(G_input, dtype=tf.float64)
        Y_target = tf.convert_to_tensor(Y_target, dtype=tf.float64)

        return [G_input, C_input_real, Y_target, C_input_gp]
