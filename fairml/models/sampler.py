"""Sampler Engine"""
import numpy as np
import tensorflow as tf



class Sampler():

	def __init__(
	    self,
	    x_idx,
	    s_idx,
	    y_idx,
	    i_idx,
	):
		tf.keras.backend.set_floatx("float64")
		self.x_idx = x_idx
		self.s_idx = s_idx
		self.y_idx = y_idx
		self.i_idx = i_idx


	def create_batches(self, dataset, batch_size):
		"""Helper Function to create batches. Each batch is a list of two samples"""

		idx = (len(dataset) // batch_size) * batch_size
		dataset = dataset.values
		np.random.shuffle(dataset)
		subset = dataset[0:idx, :]

		num_of_batches = len(dataset) // batch_size

		batches = [
		self.prepare_inputs(
			samples=self.retrieve_samples(batch, batch_size)
			)
		for batch in np.array_split(subset, num_of_batches)
		]

		return batches


	def retrieve_samples(self, batch, batch_size):
	    """Helper function to create samples from each batch"""
	    sample_size = int(batch_size / 2)
	    sample_1 = batch[0:sample_size, :]
	    sample_2 = batch[sample_size:, :]
	    return (sample_1, sample_2)


	def prepare_inputs(self, samples):
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

		return [G_input, C_input_real, Y, X]