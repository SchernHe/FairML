import numpy as np
import tensorflow as tf
from fairml.metrics.consistency import fit_nearest_neighbors

class KNNEngine():
	""" K-Neirest-Neighbors Engine"""
	def __init__(self, dataset, k,radius):
		self.dataset = dataset
		self.neigh = fit_nearest_neighbors(dataset,k)
		self.radius = radius

	def __call__(self, observations):
		neigh_dist, neigh_idx = self.neigh.radius_neighbors(observations,self.radius)
		return neigh_idx


class KNNSampler():	
	"""Sampler Engine
	
	Attributes
	----------
	dataset : pd.Dataframe
	k : int
	    Number of KNN derived total
	KNNEngine : KNNEngine
	pca_idx : list
	    Indizes of PCA columns
	random : int
	    Number of KNN sampled from total KNNs to vary NN to compare
	    each element to
	s_idx : list
	    Indizes of sensitve attributes
	x_idx : list
	    Indizes of total features
	y_idx : list
	    Indizes of Y target
	"""
	
	def __init__(
	    self,
	    dataset, 
	    x_idx,
	    s_idx,
	    y_idx,
	    pca_idx,
	    k,
	    random,
	    radius,
	):
		tf.keras.backend.set_floatx("float64")
		self.dataset = dataset.values
		self.KNNEngine = KNNEngine(self.dataset[:,pca_idx],k,radius)
		self.x_idx = x_idx
		self.s_idx = s_idx
		self.y_idx = y_idx
		self.pca_idx = pca_idx
		self.k = k
		self.random = random

	def __call__(self, batch_size: int):
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
		total_idx = range(0,len(self.dataset)-1)
		idx_choice = np.random.choice(total_idx,(len(self.dataset)//(self.k+1) ) ,replace=False)
		resulting_idx = np.delete(total_idx, idx_choice)

		num_batches = len(idx_choice)//batch_size

		batch_indizes = []
		
		for _ in range(0,num_batches):
			KNN_IDX = []
			GP_IDX = []
			observation_idx_knn = idx_choice[(_*batch_size):(_*batch_size+batch_size)]
			slice_dataset  = self.dataset[observation_idx_knn,:]
			
			# (batch_size,k=20)
			neigh_idx = self.KNNEngine(slice_dataset[:,self.pca_idx])
			for _ , idx in enumerate(neigh_idx):
				if len(idx)==0:
					print("No Neighbors Found") 
				GP_IDX.extend([idx[-1]])
				if len(idx) > self.random:
					idx = np.random.choice(idx,self.random)
				KNN_IDX.extend(idx)

			KNN_IDX = np.array(KNN_IDX)
			batch_indizes += [ (observation_idx_knn , KNN_IDX, GP_IDX) ]

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
		    Mean_C_input_real: One NN of each element of sample 1
		"""

		G_input = dataset.iloc[batch_idx[0],self.x_idx + self.s_idx].values
		C_input_real = dataset.iloc[batch_idx[1], self.y_idx].values
		Y_target = dataset.iloc[batch_idx[0], self.y_idx].values
		Mean_C_input_real = dataset.iloc[batch_idx[2], self.y_idx].values

		
		G_input = tf.convert_to_tensor(G_input,dtype=tf.float64)
		C_input_real = tf.convert_to_tensor(C_input_real,dtype=tf.float64)
		Y_target = tf.convert_to_tensor(Y_target,dtype=tf.float64)
		Mean_C_input_real = tf.convert_to_tensor(Mean_C_input_real,dtype=tf.float64)

		return [G_input, C_input_real, Y_target, Mean_C_input_real]