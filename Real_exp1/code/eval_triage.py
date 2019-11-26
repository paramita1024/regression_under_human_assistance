import time
import os
import sys 
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
from generate_data import generate_data 
from triage_human_machine import triage_human_machine


def parse_command_line_input( list_of_option, list_of_file_name ):

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, 's:l:o:f:', ['std','lamb','option', 'file_name'])

    std=0.0
    lamb=0.5
    option='greedy'
    file_name=''
    
    for opt, arg in opts:
        if opt == '-s':
            std = float(arg)
            
        if opt == '-l':
            lamb = float(arg)

        if opt == '-o':
        	for option_i in list_of_option:
        		if option_i.startswith( arg ):
        			option = option_i

        if opt == '-f':
            for file_name_i in list_of_file_name:
                #print file_name_i
                #print opt
            	if file_name_i.startswith( arg ):
            		file_name = file_name_i

        
    return std, lamb, option, file_name

class eval_triage:
	
	def __init__(self,data_file,real_flag=None, real_wt_std=None):
		self.data=load_data(data_file)
		self.real=real_flag		
		self.real_wt_std=real_wt_std


	def eval_loop(self,param,res_file,option):	
		res=load_data(res_file,'ifexists')		
		for std in param['std']:
			if self.real:
				data_dict=self.data
				triage_obj=triage_human_machine(data_dict,self.real)
			else:
				if self.real_wt_std:
					data_dict = {'X':self.data['X'],'Y':self.data['Y'],'c': self.data['c'][str(std)]}
					triage_obj=triage_human_machine(data_dict,self.real_wt_std)
				else:
					test={'X':self.data.Xtest,'Y':self.data.Ytest,'human_pred':self.data.human_pred_test[str(std)]}
					data_dict = {'test':test,'dist_mat':self.data.dist_mat,  'X':self.data.Xtrain,'Y':self.data.Ytrain,'human_pred':self.data.human_pred_train[str(std)]}
					triage_obj=triage_human_machine(data_dict,False)
			if str(std) not in res:
				res[str(std)]={}
			for K in param['K']:
				if str(K) not in res[str(std)]:
					res[str(std)][str(K)]={}
				for lamb in param['lamb']:
					if str(lamb) not in res[str(std)][str(K)]:
						res[str(std)][str(K)][str(lamb)]={}
					# res[str(std)][str(K)][str(lamb)]['greedy'] = triage_obj.algorithmic_triage({'K':K,'lamb':lamb},optim='greedy')
					print 'std-->', std, 'K--> ',K,' Lamb--> ',lamb
					res_dict = triage_obj.algorithmic_triage({'K':K,'lamb':lamb},optim=option)
					res[str(std)][str(K)][str(lamb)][option] = res_dict
					save(res,res_file)


def main():

	list_of_option =['greedy','distort_greedy','kl_triage','diff_submod']
	list_of_file_name = ['stare5','stare11','messidor', 'hatespeech'] 

	std, lamb, option, file_name = parse_command_line_input( list_of_option, list_of_file_name )
	list_of_std =[ std ]
	list_of_lamb=[ lamb ] 


	path = '../Real_Data_Results/'

	data_file = path + 'data/'+file_name

	res_file= path + file_name + '_res'
	
	for option in list_of_option:
		if option in ['diff_submod']: 
			list_of_K = [ 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 ] 
		else:
			list_of_K = [0.99]	
		param={'std':list_of_std,'K':list_of_K,'lamb':list_of_lamb}
		obj=eval_triage(data_file,real_wt_std=True)
		obj.eval_loop(param,res_file,option)	
	
if __name__=="__main__":
	main()

