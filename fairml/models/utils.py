import os 
import pandas as pd

class ResultsTracker():

	def __init__(self, path):
		self.path = path

	def write(self, results_df):
		results_path = self.path+"/Results"
		results_file = self.path + "/Results/results.csv"

		if not os.path.isdir(results_path):
			os.mkdir(results_path)

		if os.path.isfile(results_file):
		    prev_results = pd.read_csv(results_file,sep=";",index_col=0)
		    results_df = prev_results.append(results_df).reset_index(drop=True) 

		results_df.to_csv(results_file,sep=";",header=True)

	def read():
		None