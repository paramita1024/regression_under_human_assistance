import time
import matplotlib.pyplot as plt
import sys
import os
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
from generate_data import generate_data



def parse_command_line_input(  list_of_file_name ):

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, 'l:f:', ['lamb','file_name'])

    
    lamb=0.5
    file_name=''
    
    
    for opt, arg in opts:
        if opt == '-s':
            std = float(arg)
            
        if opt == '-l':
            lamb = float(arg)

        if opt == '-f':
            for file_name_i in list_of_file_name:
                #print file_name_i
                #print opt
            	if file_name_i.startswith( arg ):
            		file_name = file_name_i

        
    return lamb, file_name


class plot_triage_real:


	def __init__(self,list_of_K,list_of_std, list_of_lamb,list_of_option,list_of_test_option, flag_synthetic=None):
		self.list_of_K=list_of_K
		self.list_of_std=list_of_std
		self.list_of_lamb=list_of_lamb
		self.list_of_option=list_of_option
		self.list_of_test_option = list_of_test_option
		self.flag_synthetic = flag_synthetic



	def get_avg_error_exp2(self,res_file):
		

		option = 'greedy'
		test_method='nearest'
		res=load_data(res_file)
		
		for lamb in self.list_of_lamb:
			
			plot_obj={}
			for std in self.list_of_std:
				err_K_tr=[]
				err_K_te=[]
				for K in self.list_of_K:
					err_K_te.append(res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method]['error'])
				plot_obj[str(std)]={'test':err_K_te}
				
			self.plot_err_vs_K(plot_obj)


		


	def plot_err_vs_K(self,plot_obj):
		
		key = 'test'
		for option in plot_obj.keys():
			plt.plot( plot_obj[option][key], label=key+' '+option, linewidth=8,linestyle='--',marker='o', markersize=10)
		plt.grid()
		plt.legend()
		plt.xlabel('K')
		plt.ylabel('Average Squared Error')
		plt.title('Average Squared Error')
		plt.xticks(range(len(self.list_of_K)),self.list_of_K)
		plt.show()
	


	def get_nearest_human(self,dist,tr_human_ind):
		
		# start= time.time()
		n_tr=dist.shape[0]
		human_dist=float('inf')
		machine_dist=float('inf')
		for d,tr_ind in zip(dist,range(n_tr)):
			if tr_ind in tr_human_ind:
				if d < human_dist:
					human_dist=d
			else:
				if d < machine_dist:
					machine_dist=d
		# print 'Time required -----> ', time.time() - start , ' seconds'
		return (human_dist -machine_dist)

	def get_test_error(self,res_obj,dist_mat,x,y,y_h=None,c=None,K=None):
		
		w=res_obj['w']
		subset=res_obj['subset']
		n,tr_n=dist_mat.shape
		no_human=int((subset.shape[0]*n)/float(tr_n))

		y_m=x.dot(w)
		err_m=(y-y_m)**2
		if y_h==None:
			err_h=c  
		else: 
			err_h=(y-y_h)**2

		# start = time.time()
		diff_arr=[ self.get_nearest_human(dist,subset) for dist in dist_mat]
		# print 'Time required -----> ', time.time() - start , ' seconds'

		indices=np.argsort(np.array(diff_arr))
		subset_te_r = indices[:no_human]
		subset_machine_r=indices[no_human:]

		if subset_te_r.size==0:
			error_r =  err_m.sum()/float(n)
		else:
			error_r = ( err_h[subset_te_r].sum() + err_m.sum() - err_m[subset_te_r].sum() ) /float(n)


		subset_te_n = np.array([int(i)  for i in range(len(diff_arr)) if diff_arr[i] < 0 ])
		# print 'subset size test', subset_te_n.shape
		subset_machine_n = np.array([int(i)  for i in range(len(diff_arr)) if i not in subset_te_n ])
		# print 'sample to human--> ' , str(subset_te_n.shape[0]), ', sample to machine--> ', str( subset_machine_n.shape[0])

		if subset_te_n.size==0:
			error_n =  err_m.sum()/float(n)
		else:
			error_n = ( err_h[subset_te_n].sum() + err_m.sum() - err_m[subset_te_n].sum() ) /float(n)

		# return {'error':error, 'human_ind':subset_te, 'machine_ind':subset_machine}
		error_n={'error':error_n, 'human_ind':subset_te_n, 'machine_ind':subset_machine_n}
		error_r={'error':error_r, 'human_ind':subset_te_r, 'machine_ind':subset_machine_r}
		return error_n, error_r

	def plot_test_allocation(self,train_obj,test_obj,plot_file_path):

		x=train_obj['human']['x']
		y=train_obj['human']['y']
		plt.scatter(x,y,c='blue',label='train human')

		x=train_obj['machine']['x']
		y=train_obj['machine']['y']
		plt.scatter(x,y,c='green',label='train machine')

		x=test_obj['machine']['x'][:,0].flatten()
		y=test_obj['machine']['y']
		plt.scatter(x,y,c='yellow',label='test machine')

		x=test_obj['human']['x'][:,0].flatten()
		y=test_obj['human']['y']
		plt.scatter(x,y,c='red',label='test human')

		plt.legend()
		plt.grid()
		plt.xlabel('<-----------x------------->')
		plt.ylabel('<-----------y------------->')
		plt.savefig(plot_file_path,dpi=600, bbox_inches='tight')
		plt.close()

		# plt.show()
					
	def get_train_error(self,plt_obj,x,y,y_h=None,c=None):
		subset = plt_obj['subset']
		# print np.min(subset)
		# print np.max(subset)

		w=plt_obj['w']
		n=y.shape[0]
		if y_h==None:
			err_h=c
		else:
			err_h=(y_h-y)**2

		# print x.shape

		y_m= x.dot(w)
		err_m=(y_m-y)**2
		# print '-----------'
		# print err_h.shape
		# print err_m.shape
		# print '-----------'
		error = ( err_h[subset].sum()+err_m.sum() - err_m[subset].sum() ) /float(n)
		return {'error':error}

	def compute_result(self,res_file,data_file,option, image_file_prefix =None):
		data=load_data(data_file)
		res=load_data(res_file)
		for std,i0 in zip(self.list_of_std,range( len(self.list_of_std) )):
			for K,i1 in zip(self.list_of_K,range(len(self.list_of_K))):
				for lamb,i2 in zip(self.list_of_lamb,range(len(self.list_of_lamb))):
					if option in res[str(std)][str(K)][str(lamb)]:
						res_obj=res[str(std)][str(K)][str(lamb)][option]
						# print 'std,K,lamb,subset_size ---- > ',std,K,lamb,  res_obj['subset'].shape
						suffix='_'+option + '_std_'+str(std)+'_K_'+str(K)+'_lamb_'+str(lamb)
						image_file =  image_file_prefix + suffix #'../Synthetic_data/demo/'
						# self.plot_subset_allocation( data['X'], data['Y'], res_obj['w'], res_obj['subset'], image_file )
						# print 'std', str(std), '  K', str(K), '  lamb  ', str(lamb)
						train_res = self.get_train_error(res_obj,data['X'],data['Y'],y_h=None,c=data['c'][str(std)])
						test_res_n,test_res_r = self.get_test_error(res_obj,data['dist_mat'],data['test']['X'],data['test']['Y'],y_h=None,c=data['test']['c'][str(std)],K=K)
						res[str(std)][str(K)][str(lamb)][option]['test_res']={'ranking':test_res_r,'nearest':test_res_n}
						res[str(std)][str(K)][str(lamb)][option]['train_res']=train_res
					# else:
						
					# 	print option, ' is not evaluated for (std,K,lamb) = ', std , K , lamb
		save(res,res_file)

	def plot_subset_allocation( self, X, Y, w, subset, image_file):

		x=X[:,0].flatten()[subset]
		y=Y[subset]
		plt.scatter(x,y,c='blue',label='human')

		subset_c = np.array([ i for i in range( Y.shape[0]) if i not in  subset])
		x=X[:,0].flatten()[subset_c]
		y=Y[subset_c]
		plt.scatter(x,y,c='green',label='machine')

		
		x=X[:,0].flatten()
		y=X.dot(w)
		plt.scatter(x,y,c='yellow',label='prediction')

		plt.legend()
		plt.grid()
		plt.xlabel('<-----------x------------->')
		plt.ylabel('<-----------y------------->')
		plt.savefig(image_file+'.pdf',dpi=600, bbox_inches='tight')
		# plt.savefig(image_file+'.jpg',dpi=600, bbox_inches='tight')
		# plt.show()
		plt.close()
		# plt.show()
			
	def merge_results(self,input_res_files,merged_res_file):

		res={}
		for std in self.list_of_std:
			if str(std) not in res:
				res[str(std)]={}
			for K in self.list_of_K:
				if str(K) not in res[str(std)]:
					res[str(std)][str(K)]={}
				for lamb in self.list_of_lamb:
					if str(lamb) not in res[str(std)][str(K)]:
						res[str(std)][str(K)][str(lamb)]={}
					r=load_data(input_res_files[str(lamb)])
					# print r['0.0'].keys()
					# print res['0.0'].keys()
					res[str(std)][str(K)][str(lamb)] = r[str(std)][str(K)][str(lamb)]
		save(res,merged_res_file)

	def split_res_over_K(self,data_file,res_file,unified_K,option):
		res=load_data(res_file)
		for std in self.list_of_std:
			if str(std) not in res:
				res[str(std)]={}
			for K in self.list_of_K:
				if str(K) not in res[str(std)]:
					res[str(std)][str(K)]={}
				for lamb in self.list_of_lamb:
					if str(lamb) not in res[str(std)][str(K)]:
						res[str(std)][str(K)][str(lamb)]={}
					
					if option not in res[str(std)][str(K)][str(lamb)]:
						res[str(std)][str(K)][str(lamb)][option]={}
					if K != unified_K:
						res_dict = res[str(std)][str(unified_K)][str(lamb)][option]
						if res_dict:
							res[str(std)][str(K)][str(lamb)][option] = self.get_res_for_subset(data_file,res_dict,lamb,K)
		save(res,res_file)

	def get_optimal_pred(self,data,subset,lamb):
		
		n,dim= data['X'].shape
		subset_c=  np.array([int(i) for i in range(n) if i not in subset])	
		X_sub=data['X'][subset_c].T
		Y_sub=data['Y'][subset_c]
		subset_c_l=n-subset.shape[0]
		return LA.inv( lamb*subset_c_l*np.eye(dim) + X_sub.dot(X_sub.T) ).dot(X_sub.dot(Y_sub))

	def get_res_for_subset(self,data_file,res_dict,lamb,K):
		data=load_data(data_file)
		curr_n = int( data['X'].shape[0] * K )
		subset_tr = res_dict['subset'][:curr_n]
        	w= self.get_optimal_pred(data,subset_tr,lamb)
		return {'w':w,'subset':subset_tr}

	def set_n( self, n ):
		self.n = n




def main():

	list_of_option =['greedy']
	list_of_file_name = ['stare5','stare11'] 
	list_of_std = [0.2, 0.4, 0.6, 0.8]

	lamb, file_name = parse_command_line_input( list_of_file_name )
	list_of_lamb=[ lamb ] 

	list_of_K = [  0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 ] 
	list_of_test_option = ['nearest']

	path = '../Real_Data_Results/'

	obj=plot_triage_real(list_of_K, list_of_std, list_of_lamb, list_of_option, list_of_test_option)

	data_file = path + 'data/' + file_name 

	res_file= path + file_name + '_res'

	obj.set_n( load_data( data_file )['X'].shape[0] )
	
	for option in list_of_option:
		if option not in [ 'diff_submod']:
			unified_K = 0.99
			obj.split_res_over_K(data_file,res_file,unified_K,option)
		obj.compute_result(res_file,data_file,option, 'dummy')
	obj.get_avg_error_exp2(res_file)	


if __name__=="__main__":
	main()
