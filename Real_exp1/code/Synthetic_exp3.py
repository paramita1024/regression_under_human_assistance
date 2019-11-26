import time
import os
import sys 
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
from generate_data import generate_data 
from triage_human_machine import triage_human_machine


class Synthetic_exp3:
	def __init__(self,data_file,list_of_K, list_of_lamb, list_of_std):
		self.data=load_data(data_file)
		self.list_of_K = list_of_K
		self.list_of_lamb = list_of_lamb
		self.list_of_std = list_of_std

	def eval_loop(self, image_file_pre ):	
		for K in self.list_of_K :
			for lamb in self.list_of_lamb :
				print '\\begin{figure}[H]'		
				for std,ind  in  zip(self.list_of_std, range(len(self.list_of_std)) ):
					# suffix =  '_' + str(std) # '_'+str(K) + '_' + str(lamb) +
					image_file = image_file_pre + '_'+ str(ind ) #suffix.replace('.','_')
					caption='$K = '+str(K)+',    \\lambda='+str(lamb)+' , \\rho = '+ str(std)+'$'
					self.print_figure_singleton( image_file.split('/')[-1], caption )
					local_data = self.data[ str( std ) ]
					triage_obj=triage_human_machine( local_data, True )    
					res_obj =  triage_obj.algorithmic_triage({'K':K,'lamb':lamb},optim='greedy')
					subset_human = res_obj['subset']
					n=local_data['X'].shape[0]
					subset_machine = np.array( [ i for i in range(n) if i not in subset_human])
					self.plot_subset( local_data, res_obj['w'], subset_human,  subset_machine, image_file, K )

				print '\\caption{'+image_file_pre.split('/')[-1].replace('_',' ')+', lamb = '+str(lamb)+' }'
				print '\\end{figure}'

	def plot_subset(self, data, w, subset_human, subset_machine, image_file, K):
		x=data['X'][subset_human,0].flatten()
		y=data['Y'][subset_human]
		plt.scatter(x,y,c='red',label='human')
		self.write_to_txt( x,y, image_file+'_human')

		x=data['X'][subset_machine,0].flatten()
		y=data['Y'][subset_machine]
		plt.scatter(x,y,c='blue',label='machine')
		self.write_to_txt( x,y, image_file+'_machine')
		
		x=data['X'][:,0].flatten()
		y=data['X'].dot(w).flatten()
		plt.scatter(x,y,c='black',label='prediction')
		self.write_to_txt( x,y, image_file+'_prediction')		

		plt.legend()
		plt.grid(True)
		plt.title('Fraction of sample to human '+str( K) )
		plt.savefig(image_file + '.pdf',dpi=600, bbox_inches='tight')
		plt.close()


def main():
    list_of_std=[0.01, 0.05, 0.1, 0.5]
    list_of_lamb=[0.0001, 0.001, 0.01, 0.1 ]
    list_of_K = [0.6]	
    file_name = 'sigmoid_n_240_d_1_inc_noise'
    path = '../Synthetic_Results/'
    data_file = path + file_name
    obj=Synthetic_exp3(data_file, list_of_K, list_of_lamb, list_of_std )
    image_file_pre = path + 'Fig3/Fig3_'+file_name.split('_')[0]
    obj.eval_loop( image_file_pre)	 
	
if __name__=="__main__":
	main()

