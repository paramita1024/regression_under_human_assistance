import random
import sys
sys.path.append("/home/paramita")
import math
from myutil import *
from triage_class import * 
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
import random
import time

class triage_human_machine:
	def __init__(self,data_dict,real=None):
		self.X=data_dict['X']
		self.Y=data_dict['Y']
		# if 'test' in data_dict:
		# 	self.test=data_dict['test']
		
		if real:
			self.c = data_dict['c']
			# self.c_te = data_dict['test']['c']
		else:
			self.c=np.square(data_dict['human_pred']-data_dict['Y'])
			# self.c_te=np.square(self.test['human_pred']-self.test['Y'])
		# if 'dist_mat' in data_dict:
		# 	self.dist_mat=data_dict['dist_mat']
		
		self.dim=self.X.shape[1]
		self.n=self.X.shape[0]
		self.V=np.arange(self.n)
		self.epsilon=float(1)
		self.BIG_VALUE=100000
		self.real=real
		
	def get_c(self,subset):
		return np.array([int(i) for i in self.V if i not in subset])	
	
	def get_minus(self,subset,elm):
		return np.array([i for i in subset if i != elm] )

	def get_added(self,subset,elm):
		return np.concatenate((subset,np.array([int(elm)])),axis=0)

	def check_delete(self,g_m,subset,approx): 	
		if subset.size==0:
			# print 'subset empty'
			return False,subset
			
		g_m_subset=g_m.eval(subset)
		g_m_subset_vector=g_m.eval_vector(subset)

		if np.max(g_m_subset_vector) >= g_m_subset*approx:
			item_to_del=subset[np.argmax(g_m_subset_vector)]
			subset_left= self.get_minus(subset,item_to_del)
			# print '----Following item is deleted---',item_to_del
			# print 'now subset---> ', subset_left
			
			return True, subset_left
		# print 'Nothing deleted, subset ---> ', subset

		# print 'No deletion'
		# print 'curr function --> ',g_m_subset
		# print 'after deletion best --> ', np.max(g_m_subset_vector)
		return False, subset
		
	def check_exchange_greedy(self,g_m,subset,ground_set,approx,K):
		
		g_m_subset=g_m.eval(subset)
		g_m_exchange,subset_with_null,subset_c_gr=g_m.eval_exch_or_add(subset,ground_set,K)

		if np.max(g_m_exchange) > g_m_subset*approx:
			r,c = np.unravel_index(np.argmax(g_m_exchange,axis=None),g_m_exchange.shape)
			# print 'index of max element ',r,c
			e=subset_with_null[r]
			d=subset_c_gr[c]
			# print e,' is exchanged with ',d
			if e == -1:
				subset_with_null[r]=d
				return True,subset_with_null
			else:
				ind_e = np.where( subset == e)[0]
				# print 'subset',subset
				subset[ind_e]=d
				# print '-----------------------'
				# print subset
				# print '-----------------------'
				return True, subset
		# print 'No Exchange'
		# print 'curr function --> ',g_m_subset
		# print 'after deletion best --> ', np.max(g_m_exchange)
		return False,subset
		
	def approx_local_search(self,g_m,K,ground_set):
		# max_A (g-m)(A) given |A|<=k  	implementing local search by J.Lee 2009 STOC
		approx=1+self.epsilon/float(self.n**4)
		# print self.n
		# print self.dim
		# print '--- max elm of g-m --------'
		curr_subset=np.array([g_m.find_max_elm(ground_set)]) 
		while True:
			# start = time.time()
			# print ' ---   Delete ----- '
			flag_delete,curr_subset = self.check_delete(g_m,curr_subset,approx) 
			if flag_delete:
				# print 'deleted'
				pass
			else:
				# print ' --- Exchange ---- '
				flag_exchange,curr_subset = self.check_exchange_greedy(g_m,curr_subset,ground_set,approx,K) 
				# time.sleep(100000)
				if flag_exchange:
					pass # print 'exchanged'
				else:
					break
			# print '---------------Subset----------------------------------'
			# finish = time.time()
			# print '-----------------------------'
			# print 'Time -- > ', finish-start
			# print 'Subset', curr_subset
		return curr_subset

	def constr_submod_max_greedy(self,g_m,K) :
		# print 'constr submod max greedy'
		curr_set=np.array([]).astype(int)
		# print 'K',K
		# print 'start val---------->', g_m.eval(curr_set)
		for itr in range(K):
			
			vector,subset_left = g_m.get_inc_arr(curr_set) 
			if np.max(vector) <= 0 :
				break 
			idx_to_add=subset_left[np.argmax(vector)]
			curr_set = self.get_added(curr_set,idx_to_add)
			# print 'Iteration ',itr,'_______',g_m.eval(curr_set)
		# print 'final------------> ',g_m.eval(curr_set)
		return curr_set

	def constr_submod_max(self,g_m,K):
		
		ground_set=self.V
		# print '----- local search 1 '
		start = time.time()
		subset_1=self.approx_local_search(g_m,K,ground_set)
		ground_set=self.get_c(subset_1)
		# print '----- local search 2 '
		subset_2=self.approx_local_search(g_m,K,ground_set)
		finish  = time.time()
		print 'Time -- > ',(finish-start)
		if g_m.eval(subset_1)> g_m.eval(subset_2):  
			return subset_1
		else:
			return subset_2

	def sel_subset_diff_submod_greedy(self):
		# solve difference of submodular functions
		subset_old=np.array([])
		g_f=G({'X':self.X,'Y':self.Y,'c':self.c,'lamb':self.lamb})
		val_old = g_f.eval(subset_old)
		# print 'VAl---------------> ',val_old
		itr=0
		while True :
			# print '-----Diff submod greedy----------Iter ', itr, '  ---------------------------------------'
			# print 'modular upper bound '
			# return 
			f=F({'X':self.X,'Y':self.Y,'c':self.c,'lamb':self.lamb})
			m_f=f.modular_upper_bound(subset_old)
			g_m=SubMod({'X':self.X,'lamb':self.lamb,'m':m_f})
			subset=self.constr_submod_max_greedy(g_m,self.K) 

			# check whether g-f really improve 
			val_curr = g_f.eval(subset)	
			# print 'VAl---------------> ',val_curr
			if val_curr <= val_old:
				return subset_old
			if set(subset) == set(subset_old) :
				return subset
			else:
				subset_old=subset
				val_old=val_curr
			itr += 1
		
	def sel_subset_diff_submod(self):	
		# solve difference of submodular functions
		subset_old=np.array([])

		itr=0
		while True :
			# print '-------------------------------Iter ', itr, '  ---------------------------------------'
			# print 'modular upper bound '
			f=F({'X':self.X,'Y':self.Y,'c':self.c,'lamb':self.lamb})
			m_f=f.modular_upper_bound(subset_old)
			# m_f=self.modular_upper_bound(subset_old) 
			# print 'constr submodular max '
			g_m=SubMod({'X':self.X,'lamb':self.lamb,'m':m_f})
			subset=self.constr_submod_max(g_m,self.K) 
			# print '--------OLD-------------------'
			print 'subset length', subset.shape
			# print '---------New-------------------'
			# print subset
			if set(subset) == set(subset_old) :
				return subset
			else:
				subset_old=subset
			itr += 1
		
	def set_param(self,lamb,K):
		self.lamb=lamb 
		self.K=K 
	
	def get_optimal_pred(self,subset):
		subset_c=self.get_c(subset)
		X_sub=self.X[subset_c].T
		Y_sub=self.Y[subset_c]
		subset_c_l=self.n-subset.shape[0]
		return LA.inv( self.lamb*subset_c_l*np.eye(self.dim) + X_sub.dot(X_sub.T) ).dot(X_sub.dot(Y_sub))
		
	def plot_subset(self,w,subset,K):
		plt_obj={}

		x=self.X[subset,0].flatten()
		y=self.Y[subset]
		# plt.scatter(x,y,c='red',label='human')
		plt_obj['human']={'x':x,'y':y}


		c_subset = self.get_c(subset)
		x=self.X[c_subset,0].flatten()
		y=self.Y[c_subset]
		# plt.scatter(x,y,c='blue',label='machine')
		plt_obj['machine']={'x':x,'y':y}
		
		x=self.X[:,0].flatten()
		y=self.X.dot(w).flatten()
		plt_obj['prediction']={'x':x,'y':y,'w':w}
		# plt.scatter(x,y,c='black',label='prediction')		
		# plt.ylim([-1,1])
		# plt.legend()
		# plt.grid(True)
		# plt.title('Fraction of sample to human'+str(K) )
		# plt.show()	
		return plt_obj

	def distort_greedy(self,g,K,gamma):

		c_mod = modular_distort_greedy({'X':self.X,'Y':self.Y,'c':self.c,'lamb':self.lamb}) 
		subset=np.array([]).astype(int)
		g.reset()
		for itr in range(K):
			frac = (1-gamma/float(K) )**(K-itr-1)
			# print frac
			subset_c = self.get_c(subset)
			# print subset_c.shape
			c_mod_inc = c_mod.get_inc_arr(subset).flatten()
			# print 'c',c_mod_inc.shape
			g_inc_arr, subset_c_ret =  g.get_inc_arr(subset)
			g_pos_inc  = g_inc_arr.flatten() + c_mod_inc
			inc_vec = frac * g_pos_inc - c_mod_inc
			# print '------------'
			# print subset_c.shape
			# print inc_vec.shape
			# print '------------' 
			if np.max( inc_vec ) <= 0 :
				print 'no increment'
				return subset
			sel_ind = np.argmax(inc_vec)	
			elm = subset_c[sel_ind]
			subset = self.get_added( subset, elm)
			g.update_data_str(elm)
		return subset

	def stochastic_distort_greedy(self,g,K,gamma,epsilon):
		c_mod = modular_distort_greedy({'X':self.X,'Y':self.Y,'c':self.c,'lamb':self.lamb}) 
		subset=np.array([]).astype(int)
		g.reset()
		s=int(math.ceil(self.n*np.log( float(1)/epsilon ) / float(K) ) )
		print 'subset_size', s, 'K-->', K, ', n --> ', self.n
		for itr in range(K):
			frac = (1-gamma/float(K) )**(K-itr-1) 
			subset_c = self.get_c(subset)
			if s < subset_c.shape[0]:
				subset_choosen = np.array( random.sample( subset_c, s ) )
			else:
				subset_choosen = subset_c

			c_mod_inc = c_mod.get_inc_arr( subset, rest_flag = True, subset_rest=subset_choosen ) 
			g_inc_arr, subset_c_ret =  g.get_inc_arr( subset, rest_flag=True, subset_rest = subset_choosen )
			g_pos_inc  = g_inc_arr + c_mod_inc
			inc_vec = frac * g_pos_inc - c_mod_inc
			if np.max( inc_vec ) <= 0 :
				return subset
			sel_ind = np.argmax(inc_vec)	
			elm = subset_choosen[sel_ind]
			subset = self.get_added( subset, elm)
			g.update_data_str(elm)
		return subset

	def gamma_sweep_distort_greedy(self, flag_stochastic=None):
		g=G({'X':self.X,'Y':self.Y,'c':self.c,'lamb':self.lamb})
		# Submod_ratio = Submodularity_ratio({'X':self.X,'Y':self.Y,'c':self.c,'lamb':self.lamb}) 
		delta = 0.01
		# arr = np.array([int(math.ceil( (1/delta)* np.log( 1/max(delta,Submod_ratio) ))) for Submod_ratio in [.5,.6,.7,.8,.9]])
		Submod_ratio = 0.7
		#----------------------------------CHANGE **********************************
		T=5#10 #  int(math.ceil( (1/delta)* np.log( 1/max(delta,Submod_ratio) ))) # check 
		subset = {}
		G_subset=[]
		gamma = 1.0
		# print 'T',T
		# start = time.time()
		for r in range(T+1): 
			# print r
			if flag_stochastic:
				subset_sel = self.stochastic_distort_greedy(g,self.K,gamma,delta) 
			else:
				subset_sel = self.distort_greedy(g,self.K,gamma) 
			subset[str(r)] = subset_sel
			# print 'itr-->', r , 'subset size -- > ', subset_sel.shape[0]
			G_subset.append( g.eval(subset_sel))
			gamma = gamma*(1-delta)
			# print time.time() - start
		empty_set = np.array([]).astype(int)
		subset[str(T+1)]=empty_set
		G_subset.append( g.eval(empty_set))
		# plt.plot(np.squeeze(G_subset) )
		# plt.show()
		max_set_ind = np.argmax( np.array(G_subset))
		return subset[ str( max_set_ind ) ]

	def max_submod_greedy(self):

		curr_set=np.array([]).astype(int)
		g=G({'X':self.X,'Y':self.Y,'c':self.c,'lamb':self.lamb})
		# print 'Need to select ', self.K , ' items'
		start = time.time()	
		for itr in range(self.K):
			vector,subset_left = g.get_inc_arr(curr_set)
			# print 'inc',np.max(vector)
			# if np.max(vector) < 0 :
			# 	# print 'dataset size',curr_set.shape[0]
			# 	return curr_set
			idx_to_add=subset_left[np.argmax(vector)]
			curr_set = self.get_added(curr_set,idx_to_add)
			g.update_data_str(idx_to_add)
		
			if itr % 50 == 0 :
				time_r = time.time() - start
				# print 'itr ', itr
				# print time_r, ' seconds'
		# print 'dataset size',curr_set.shape[0]
		return curr_set

	def kl_triage_subset(self):
		kl_obj = kl_triage({'X':self.X,'Y':self.Y,'c':self.c,'lamb':self.lamb})
		tmp = kl_obj.get_subset(self.K)
		# print 'K', self.K
		# print 'self.n', self.n
		# print 'subset', tmp.shape
		return  kl_obj.get_subset(self.K)

	def algorithmic_triage(self,param,optim):
		# start=time.time()
		# print 'check',K#int(K*self.n)
		self.set_param(param['lamb'],int(param['K']*self.n))
		
		if optim=='diff_submod':
			subset  = self.sel_subset_diff_submod() 
		if optim == 'greedy':
			subset=self.max_submod_greedy()		
		if optim == 'diff_submod_greedy':
			subset  = self.sel_subset_diff_submod_greedy() 
		if optim == 'distort_greedy':
			subset = self.gamma_sweep_distort_greedy(flag_stochastic=False)
		if optim == 'kl_triage':
			subset = self.kl_triage_subset()
		if optim == 'stochastic_distort_greedy':
			subset = self.gamma_sweep_distort_greedy(flag_stochastic=True)
		# print 'subset_size', subset.shape
		if subset.shape[0] == self.n:
			w_m = 0 
		else:
			w_m = self.get_optimal_pred(subset)
		
		plt_obj={'w':w_m,'subset':subset}
		return plt_obj

	
	# def get_avg_accuracy(self,w,subset,nbr):
		
	# 	predict=(self.Y[subset]-self.X[subset].dot(w)).flatten()
	# 	error = ( predict.dot(predict) + self.c.sum()-self.c[subset].sum())/self.n
	# 	subset_te=[]
	# 	for dist in self.dist_mat:
	# 		indices = np.argsort(dist)[:nbr]
	# 		dist_elm= dist[indices]
	# 		indicator = np.array([1 if i in subset else -1 for i in indices])
	# 		if dist_elm.dot(indicator) > 0 :
	# 			subset_te.append(1)
	# 		else:
	# 			subset_te.append(0)
	# 	subset_te=np.array(subset_te,dtype=bool)

	# 	predict_te=(self.test['Y'][ subset_te ] -self.test['X'][ subset_te ].dot(w)).flatten()
	# 	error_te = (predict_te.dot(predict_te) + self.c_te.sum()-self.c_te[subset_te].sum())/self.test['Y'].shape[0]


	# 	res={'avg_train_err':error,'avg_test_err':error_te}
	# 	return res 


	# def check_exchange_incremental(self,subset,m,ground_set,approx,K):
		
		
	# 	l_subset=subset.shape[0]
	# 	subset_c=self.get_c(subset)
	# 	X_sub = self.X[subset_c].T
	# 	A=X_sub.dot(X_sub.T)
	# 	B=self.addend( l_subset,'g')
	# 	if subset.size == 0:
	# 		g_m_subset=np.log( LA.det(A+B))
	# 	else:
	# 		g_m_subset=np.log( LA.det(A+B))  - m[subset].sum()
		
	# 	if subset.shape[0]<K:
	# 		subset_with_null = np.hstack((subset,np.array([-1])))
			
	# 	print 'g_m_subset---------------->',g_m_subset
	# 	for e in subset_with_null : 
	# 		for d in [ i for i in ground_set if i not in subset]:
	# 			if e == -1:		
	# 				g_part=np.log(LA.det(A-self.elm_mat(d,'g')+self.addend(l_subset+1,'g')  ))
	# 				if subset.size==0:
	# 					m_part = m[d]
	# 				else:
	# 					m_part=m[subset].sum()+m[d]
	# 			else:
	# 				g_part=np.log(LA.det(A+self.elm_mat(e,'g')-self.elm_mat(d,'g')+B))
	# 				if subset.size==0:
	# 					m_part = m[subset].sum()-m[e]+m[d]
	# 			g_m_new = g_part+m_part
	# 			print 'g_m_new----------------->',g_m_new
	# 			if g_m_new >= g_m_subset: # *approx: *******************
	# 				print 'e------------->',e,', d--------------> ',d
					
	# 				if e == -1:
						
	# 					return  np.array(( subset , np.array([d]) ))
	# 				else:
						
	# 					ind_e = np.where( subset == e)[0]
	# 					print 'subset',subset
	# 					subset[ind_e]=d
	# 					print '-----------------------'
	# 					print subset
	# 					print '-----------------------'
	# 					return True, subset
					
	# 	return False,subset
# def get_avg_accuracy(self,w,subset,nbr):
		
	# 	predict=(self.Y[subset]-self.X[subset].dot(w)).flatten()
	# 	error = ( predict.dot(predict) + self.c.sum()-self.c[subset].sum())/self.n
	# 	subset_te=[]
	# 	for dist in self.dist_mat:
	# 		indices = np.argsort(dist)[:nbr]
	# 		dist_elm= dist[indices]
	# 		indicator = np.array([1 if i in subset else -1 for i in indices])
	# 		if dist_elm.dot(indicator) > 0 :
	# 			subset_te.append(1)
	# 		else:
	# 			subset_te.append(0)
	# 	subset_te=np.array(subset_te,dtype=bool)

	# 	predict_te=(self.test['Y'][ subset_te ] -self.test['X'][ subset_te ].dot(w)).flatten()
	# 	error_te = (predict_te.dot(predict_te) + self.c_te.sum()-self.c_te[subset_te].sum())/self.test['Y'].shape[0]


	# 	res={'avg_train_err':error,'avg_test_err':error_te}
	# 	return res 
