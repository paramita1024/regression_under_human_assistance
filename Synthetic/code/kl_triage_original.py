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
class kl_triage:
    def __init__(self,data):
        self.X=data['X']
        self.Y=data['Y']
        self.c=data['c']
        self.test = data['test']
        self.lamb=data['lamb']
        self.n,self.dim = self.X.shape
        self.V = np.arange(self.n)
        self.training()
        
    def	training(self):
        
        self.train_machine_error()
        self.train_human_error()
        
    def testing(self, K_frac):
        
        n= self.test['X'].shape[0]
        K = int( n*K_frac)

        # print 'w machine err', self.w_machine_error.shape
        # print 'w_human_err' , self.w_human_error.shape

        machine_err = self.test['X'].dot( self.w_machine_error )
        human_err = self.test['X'].dot( self.w_human_error )

        err = human_err - machine_err
        indices = np.argsort( err )

        human_indices = indices[ : K ]
        machine_indices = indices[ K : ]


        y_m = self.test['X'].dot( self.w_machine_pred )
        
        test_err  = ( ((self.test['Y']-y_m)**2)[machine_indices].sum()  + self.test['c'][human_indices].sum())/float( n )
        return test_err 

    def train_machine_error( self ):

        self.w_machine_pred, tr_err = self.fit_LR( self.X, self.Y)
        self.w_machine_error, tmp = self.fit_LR( self.X, tr_err)

    def train_human_error( self ):

        self.w_human_error, tmp = self.fit_LR( self.X, np.sqrt(self.c) )

    def fit_LR(self, X, Y) :

        w = LA.solve( X.T.dot(X)+ self.lamb * np.eye( X.shape[1] ), X.T.dot(Y) ) 
        err = np.absolute((X.dot(w)-Y))
        return w,err




    
