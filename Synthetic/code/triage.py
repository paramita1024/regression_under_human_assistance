from myutil import *
#from sets import Set
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
import random
				
class triage:
	def __init__(self,data_dict):
		self.X=data_dict['X']
		self.Y=data_dict['Y']
		self.test=data_dict['test']
		self.dist_mat=data_dict['dist_mat']
		self.c=np.square(data_dict['human_pred']-data_dict['Y'])
		self.c_te=np.square(self.test['human_pred']-self.test['Y'])
		self.dim=self.X.shape[1]
		self.n=self.X.shape[0]
		# print 'n',self.n
		self.V=np.arange(self.n)

		
	def get_single_cov_mat(self,elm):
		return self.X[elm].reshape(self.dim,1).dot(self.X[elm].reshape(1,self.dim))

	def modular_approx(self, perm):
		buffer_var=np.zeros((self.dim,self.dim))
		h=np.zeros(self.n)
		prev_val=0
		for elm in perm:
			buffer_var+=(self.lamb*np.eye(self.dim)+self.get_single_cov_mat(int(elm)))
			curr_val=np.log(LA.det(buffer_var))
			h[elm]=curr_val - prev_val
			prev_val=curr_val
		return h
			

	def get_curr_mat(self,elm):
		curr_vector=np.vstack((self.Y[elm],self.X[elm].reshape(self.dim,1)))
		curr_mat = curr_vector.dot(curr_vector.T)
		curr_mat[1:,1:] += self.lamb*np.eye(self.dim)
		curr_mat[0,0] -= self.c[elm]
		return curr_mat

	def to_be_added(self,flag,subset):
		first_row=np.zeros((1,self.dim+1))
		first_col=np.zeros((self.dim,1))
		if flag:
			first_row[0]+=(np.sum(self.c)-np.sum(self.c[subset]))
			corner_mat=self.lamb*subset.shape[0]*np.eye(self.dim)
		else:
			corner_mat=self.lamb*self.n*np.eye(self.dim)
		addend_mat=np.vstack((first_row ,np.hstack(( first_col,corner_mat ))  ))
		return addend_mat

	def get_modular_upper_bound(self,subset,h):
			
		Y_X=np.concatenate((self.Y.reshape(1,self.n),self.X.T),axis=0)
		buffer_V = Y_X.dot(Y_X.T)+self.to_be_added(False,subset)
		f_V=np.log(LA.det(buffer_V))
		
		buffer_subset = Y_X[:,subset].dot(Y_X[:,subset].T)+ self.to_be_added(True,subset)
		f_subset=np.log(LA.det(buffer_subset))

		g=np.zeros(self.n)
		for elm in self.V:
			if elm in subset:
				curr_mat = self.get_curr_mat(elm) 
				buffer_curr = buffer_V-curr_mat
				g[elm]=f_V - np.log(LA.det(buffer_curr))
			else:
				curr_mat = self.get_curr_mat(elm) 
				buffer_curr = buffer_subset+curr_mat
				g[elm]=np.log(LA.det(buffer_curr))-f_subset
		return (g-h)

		

	def get_lowest_entries(self,g,K):
		return np.argpartition(g,K)[:K]


	def constr_submod_min(self,h):
		# min_A f(A) given |A|>=k , 	
		target=np.copy(self.V)
		# print '----------'
		# print target
		rand.shuffle(target)
		target = target[:self.K]
		# print target
#		print target.shape[0]
		while True:
			g=self.get_modular_upper_bound(target,h)
			new_target=self.get_lowest_entries(g,self.K)
			if len( set(target).symmetric_difference(set(new_target) ))==0:
				break
			else:
				target=new_target
#			print target.shape[0]
		return target

	def gen_random_perm(self,target):
		prefix_part=np.copy(target)
		suffix_part=np.array([i for i in self.V if i not in target ])
		rand.shuffle(prefix_part)
		rand.shuffle(suffix_part)
		return np.hstack((prefix_part,suffix_part))


	def get_func(self,subset):
		subset_len=subset.shape[0]
		#		print subset_len
		X_sub=self.X[subset].T
		Y_sub=self.Y[subset]
		X_Y_sub=(X_sub.dot(Y_sub)).flatten()
		#      print X_Y_sub.shape
		X_X=LA.inv( self.lamb*subset_len*np.eye(self.dim)+  X_sub.dot(X_sub.T) )
		f_m=Y_sub.dot(Y_sub)- X_Y_sub.dot( X_X.dot(X_Y_sub) )
		f_h=self.c.sum()-self.c[subset].sum()
		return np.log(f_m+f_h)



	def sel_subset_diff_submod(self,delta):	
		# solve difference of submodular functions
		curr_perm=np.copy(self.V)
		rand.shuffle(curr_perm)	
		min_val= float('inf')
		while True :
			h=self.modular_approx(curr_perm) 
			target=self.constr_submod_min(h) #
			curr_val=self.get_func(target) 
			curr_perm=self.gen_random_perm(target)
			if curr_val<min_val - delta:
				min_val =curr_val
			else:
				break
		return target	
		

	def set_param(self,lamb,K):
		
		self.lamb=lamb 
		self.K=K 
		

	def get_avg_accuracy(self,w,subset,nbr):
		
		predict=(self.Y[subset]-self.X[subset].dot(w)).flatten()
		error = ( predict.dot(predict) + self.c.sum()-self.c[subset].sum())/self.n


		subset_te=[]
		for dist in self.dist_mat:
			indices = np.argsort(dist)[:nbr]
			dist_elm= dist[indices]
			indicator = np.array([1 if i in subset else -1 for i in indices])
			if dist_elm.dot(indicator) > 0 :
				subset_te.append(1)
			else:
				subset_te.append(0)
		subset_te=np.array(subset_te,dtype=bool)

		predict_te=(self.test['Y'][ subset_te ] -self.test['X'][ subset_te ].dot(w)).flatten()
		error_te = (predict_te.dot(predict_te) + self.c_te.sum()-self.c_te[subset_te].sum())/self.test['Y'].shape[0]


		res={'avg_train_err':error,'avg_test_err':error_te}
		return res 

	def get_optimal_pred(self,subset):
		
		X_sub=np.hstack((self.X[subset], np.ones((subset.shape[0],1))  )).T
#		X_sub=self.X[subset].T
		Y_sub=self.Y[subset]
		subset_l=subset.shape[0]
		return LA.inv( self.lamb*subset_l*np.eye(self.dim)+ X_sub.dot(X_sub.T) ).dot(X_sub.dot(Y_sub))
		

	def algorithmic_triage(self,param,num_nbr):
		# print 'check',K#int(K*self.n)
		self.set_param(param['lamb'],int(param['K']*self.n))
		subset_for_machine  = self.sel_subset_diff_submod(param['delta'])
		w_m = self.get_optimal_pred(subset_for_machine)
		res_dict=self.get_avg_accuracy(w_m, subset_for_machine,num_nbr)
		return res_dict
		
		
	def plot_subset(self,w,subset,K):
		x=self.X[subset]
		y=self.Y[subset]
		plt.scatter(x,y,c='red',label='machine')
		
		c_subset = np.array([i for i in self.V if i not in subset])
		x=self.X[c_subset]
		y=self.Y[c_subset]
		plt.scatter(x,y,c='blue',label='human')
		
		
		x=self.X
#		print w.shape
		y=np.hstack(( self.X, np.ones((self.n,1)) )).dot(w.reshape(2,1))
		
		plt.scatter(x,y,c='black',label='prediction')		
		
		plt.legend()
		plt.grid(True)
		plt.ylim([-2,2])
		plt.title('Fraction of sample to human'+str(1-K) )
		
		plt.show()		
		
		
		
	def algorithmic_triage_visualize_train(self,param):
		self.set_param(param['lamb'],int(param['K']*self.n))
		subset_for_machine  = self.sel_subset_diff_submod(param['delta'])
		w_m=self.get_optimal_pred(subset_for_machine)
		self.plot_subset(w_m,subset_for_machine,param['K'])
#		res_dict=self.get_avg_accuracy(w_m, subset_for_machine,num_nbr)
		return {} # res_dict
		



