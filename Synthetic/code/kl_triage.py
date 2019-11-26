import sys
sys.path.append("/home/paramita")
import math
from myutil import *
# from triage_class import * 
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
        self.lamb=data['lamb']
        self.n,self.dim = self.X.shape
        self.V = np.arange(self.n)
        self.training()
        
    def	training(self):
        
        self.train_machine_error()
        self.train_human_error()

    def get_subset(self,K):

        machine_err = self.X.dot( self.w_machine_error )
        human_err = self.X.dot( self.w_human_error )

        err = human_err - machine_err
        indices = np.argsort( err )
        
        return indices[:K]
        
    def train_machine_error( self ):

        self.w_machine_pred, tr_err = self.fit_LR( self.X, self.Y)
        self.w_machine_error, tmp = self.fit_LR( self.X, tr_err)

    def train_human_error( self ):

        self.w_human_error, tmp = self.fit_LR( self.X, np.sqrt(self.c) )

    def fit_LR(self, X, Y) :

        w = LA.solve( X.T.dot(X)+ self.lamb * np.eye( X.shape[1] ), X.T.dot(Y) ) 
        err = np.absolute((X.dot(w)-Y))
        return w,err
