import sys
from os import listdir
from os.path import isfile, join
import numpy.random as rand
import codecs
import csv 
import random
import fasttext
from myutil import *
import numpy as np
import numpy.linalg as LA
from scipy.io import arff
import shutil 
from PIL import Image
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from real_data_classes import *


class Generate_human_error:
    
    def __init__(self, data_file):
        self.data = load_data( data_file )
        if 'c' in self.data:
            del self.data['c']
            del self.data['test']['c']
        self.n, self.dim  = self.data['X'].shape

    def normalize_features(self, delta = 1 ):
        
        n,dim = self.data['X'].shape
        for feature in range( dim ):
            self.data['X'][:,feature] = np.true_divide( self.data['X'][:,feature], 100*LA.norm( self.data['X'][:,feature].flatten() ) )
            self.data['test']['X'][:,feature] = np.true_divide( self.data['test']['X'][:,feature], 100*LA.norm( self.data['test']['X'][:,feature].flatten() ) )
        
        print np.max( [ LA.norm(x.flatten()) for x in self.data['X']] )

        

    def vary_discrete( self, noise_ratio, list_of_high_err_const):
        def get_num_category( y, y_t):
            y = np.concatenate(( y.flatten(), y_t.flatten() ), axis = 0 )
            return np.unique( y ).shape[0] 

        def nearest( i ):
            return np.argmin( self.data['dist_mat'][i])   

        num_cat = get_num_category( self.data['Y'], self.data['test']['Y'])
        n=self.data['X'].shape[0]
        indices = np.arange( n )
        random.shuffle(indices)
        # err =  (( float(1)/num_cat )**2 )/20 
        self.data['low']={}
        self.data['c']={}
        self.data['test']['c']={}
        
        for high_err_const in list_of_high_err_const:
            num_low = int(high_err_const*n)
            self.data['low'][str(high_err_const)]=indices[:num_low]
            # self.data['c'][str(high_err_const)] = np.array( [ err if i in self.data['low'][str(high_err_const)] else high_err_const*err for i in range(n) ] )
            self.data['c'][str(high_err_const)] = np.array( [ 0.0001 if i in self.data['low'][str(high_err_const)] else 0.25 for i in range(n) ] )
            self.data['test']['c'][str(high_err_const)] = np.array( [ 0.0001 if nearest(i) in self.data['low'][str(high_err_const)] else 0.5 for i in range( self.data['test']['X'].shape[0]) ] )
        
    def save_data(self, data_file):
        save( self.data , data_file)

def generate_human_error( path, file_name_list):

    list_of_std = [0.2, 0.4, 0.6, 0.8]

    for file_name in file_name_list:

        data_file = path + 'data/' + file_name        
        obj = Generate_human_error( data_file )
        obj.vary_discrete( list_of_std )
        obj.save_data( data_file )
    
def main():
     
    file_name_list = ['stare5','stare11'] 
    path = '../Real_Data_Results/'
    generate_human_error( path , file_name_list )
    
if __name__=="__main__":
	main()

        
    # def vary_discrete_3( self, list_of_noise_ratio ):
    #     def get_num_category( y, y_t):
    #         y = np.concatenate(( y.flatten(), y_t.flatten() ), axis = 0 )
    #         return np.unique( y ).shape[0] 
        
    #     def nearest( i ):
    #         return np.argmin( self.data['dist_mat'][i])   

    #     self.normalize_features()

    #     num_cat = get_num_category( self.data['Y'], self.data['test']['Y']) 
    #     # print num_cat
    #     # print float(1)/9
    #     # return        
    #     high_err_const = 45
    #     n=self.data['X'].shape[0]
    #     indices = np.arange( n )
    #     random.shuffle(indices)
    #     err =  (( float(1)/num_cat )**2 )/50 
    #     print err
    #     print 45*err
    #     # return 
    #     self.data['low']={}
    #     self.data['c']={}
    #     self.data['test']['c']={}
        

    #     # list_of_low_indices = []
    #     for noise_ratio in list_of_noise_ratio:
    #         num_low = int(noise_ratio*n)
    #         # print num_low
    #         self.data['low'][str(noise_ratio)]=indices[:num_low]
    #         self.data['c'][str(noise_ratio)] = np.array( [ err if i in self.data['low'][str(noise_ratio)] else high_err_const*err for i in range(n) ] )
    #         print 'c min', np.min( self.data['c'][str( noise_ratio )])
    #         print 'c max', np.max( self.data['c'][str( noise_ratio )])
    #         self.data['test']['c'][str(noise_ratio)] = np.array( [ err if nearest(i) in self.data['low'][str(noise_ratio)] else high_err_const*err for i in range( self.data['test']['X'].shape[0]) ] )
        
