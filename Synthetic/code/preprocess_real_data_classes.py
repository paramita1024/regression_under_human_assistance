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

def split_data(frac,file_data,file_data_split):
    
    data = load_data(file_data)

    print 'x' , data['x'].shape 
    print 'y' , data['y'].shape
    for key in data['c'].keys():
        print 'c' , data['c'][key].shape

    num_data=data['y'].shape[0]
    num_train=int(frac*num_data)
    num_test=num_data-num_train 
    indices=np.arange(num_data)
    random.shuffle(indices)
    indices_train=indices[:num_train]
    indices_test=indices[num_train:]
    data_split={}
    data_split['X']=data['x'][indices_train]
    data_split['Y']=data['y'][indices_train]
    test={}
    test['X']=data['x'][indices_test]
    test['Y']=data['y'][indices_test]
    data_split['c']={}
    test['c']={}
    list_of_std = data['c'].keys()
    for std in list_of_std:
        data_split['c'][str(std)]=data['c'][str(std)][indices_train]
        test['c'][str(std)]=data['c'][str(std)][indices_test]
    data_split['test']=test
    data_split['dist_mat']=np.zeros((num_test,num_train))
    for te in range(num_test):
        for tr in range(num_train):
            data_split['dist_mat'][te,tr]=LA.norm( test['X'][te] - data_split['X'][tr] )
    save( data_split, file_data_split)

def process_data(data_file, data_pca_file, num_features ):

    data = load_data(data_file)
    sc = StandardScaler() 
    pca = PCA(n_components = num_features) 

    X_train = sc.fit_transform(data['X']) 
    X_test = sc.transform(data['test']['X']) 

    X_train = pca.fit_transform(X_train) 
    X_test = pca.transform(X_test) 

    data['X']=X_train
    data['test']['X']=X_test
    save( data, data_pca_file )

def write_file_to_txt(  data_file_pca, data_file_txt, annotations_file, image_vector_file ):
     
    annotations = load_data( annotations_file )
    image_vector_dict = load_data( image_vector_file )
    n = image_vector_dict['names'].shape[0]
    annote = np.zeros((n,20))
    print annotations.keys()
    for name,i in zip(image_vector_dict['names'], range(n) ) :
        print i
        print name
        key = name.split('/')[-1][:-4]
        if 'CheXpert'  in data_file_pca:
            key = key + '.jpg'
        annote[i] = annotations[ key ]
        if 'EyePAC' in data_file_pca:
            if i==749:
                n=750
                break
    num_tr = int(0.8*n)
    annote_tr = annote[:num_tr] 
    annote_te = annote[num_tr:]
    data = load_data(data_file_pca)
    with open(data_file_txt+'_train.txt','w' ) as f:
        for i in range(num_tr):
            f.write(','.join( map(str, data['X'][i]) ) + ',')
            f.write(','.join( map(str, annote_tr[i]) ) + ',' )
            f.write( str(data['Y'][i])  + '\n')
    with open(data_file_txt+'_test.txt','w' ) as f:
        for i in range(n-num_tr):
            f.write(','.join( map(str, data['test']['X'][i]) ) + ',')
            f.write(','.join( map(str, annote_te[i]) ) + ',' )
            f.write( str(data['test']['Y'][i])  + '\n')


class CheXpert:
    def __init__(self, path ):
        self.path = path 
        label_file =  self.path + 'CheXpert-v1.0-small/train.csv'
        label_dict_file = self.path + 'label_dict'
        self.selected_cols =[ str(i) for i in [7,8,13,14,15,16] ] 
        # self.read_label(label_file, label_dict_file)
        # self.create_sep_dir( label_dict_file )
        for i in  self.selected_cols:
            label_file = self.path + 'data/' + i + '/label'
            image_vector_file = self.path + 'data/' + i + '/out_CheXpert_' + i 
            data_file = self.path + 'data/' + i + '/data'
            annote_file= self.path + 'data/' + i + '/annote'
            # self.map_label( label_file )
            # self.read_image_vector(image_vector_file, label_file, data_file, annote_file)
            data_file_split = self.path + 'data/' + i + '/data_split'
            data_file_pca = data_file_split + '_pca'
            # split_data( 0.8, data_file, data_file_split )
            # process_data( data_file_split, data_file_pca, num_features=100  )
            data_file_txt = '../Real_Data/text_files/'+ self.path.split('/')[2]+'_'+i
            write_file_to_txt(  data_file_pca, data_file_txt, annote_file, image_vector_file )

    def map_label( self, label_file):

        label_dict = load_data( label_file)
        for key in label_dict.keys():
            val = label_dict[key]
            label_dict[key] = int(val+1)
        save( label_dict, label_file)

    def read_label(self, label_file, label_dict_file): 
        
        cols = { dir:0 for dir in self.selected_cols }
        dict_label={}
        for i in cols.keys():
            dict_label[i]={}
        rows = [] 
        count = 1
        with open(label_file, 'r') as csvfile: 
            csvreader = csv.reader(csvfile)
            fields = csvreader.next() 
            for row in csvreader:
                count+=1
                # print '------'
                # print row
                # print('-'*50)
                name = row[0]
                # name_suffix = '_'.join(name.split('/')[2:])
                for i in cols.keys():
                    if row[int(i)]:
                        dict_label[i][name] = float(row[int(i)])
                        cols[i] += 1
                        if cols[i] == 1000:
                            del cols[i]
                            if not cols:
                                print count 
                                save( dict_label, label_dict_file)
                                return
                # j+=1
                # if j==100:
                #     print cols
                #     print dict_label
                #     return  
        save(dict_label,label_dict_file)

    def create_sep_dir(self, label_dict_file ):

        # for i in self.selected_cols:
            # os.mkdir( self.path + 'data/' + i)
            # os.mkdir( self.path + 'data/' + i + '/Images')
        # return 
        label_dict = load_data(label_dict_file )
        for dir in self.selected_cols:
            label_dict_dir = {}
            for name in label_dict[dir].keys():
                name_suffix = '_'.join(name.split('/')[2:])
                src = self.path + name 
                dest = self.path + 'data/'+ str(dir) + '/Images/' + name_suffix
                shutil.copy( src, dest )
                label_dict_dir[ name_suffix ] = label_dict[ str(dir) ][name]
            label_dict_dir_file = self.path + 'data/' + dir + '/label'
            save( label_dict_dir ,  label_dict_dir_file )

    def read_image_vector(self, image_file, label_file, data_file, annote_file ):
        image_vec_dict = load_data(image_file)
        label_dict = load_data(label_file)
        annote_dict = {}
        x=[]
        y=[]
        c=[]
        self.annote = human_annotation( 0,2,0,1,20,mode='auto',data=label_dict.values() )
        for name,vector in zip(image_vec_dict['names'], image_vec_dict['data']  ):
            # print name 
            # return 
            key = name.split('/')[-1]
            # print key 
            # return 
            x.append(vector)
            y_ground = label_dict[key]
            y.append( self.annote.map_range(y_ground )  )
            annotation = self.annote.annote_it( y_ground )
            c.append( self.annote.get_error( annotation , y_ground ) )
            annote_dict[ key ] = annotation
            
        save({'x':np.array(x), 'y':np.array(y), 'c':{'0.0':np.array(c)} }, data_file )
        save( annote_dict, annote_file )
    
class human_annotation:
    def __init__(self,l,h,l_new,h_new,num_annote, mode = None, data=None):
        self.l=l
        self.h=h
        self.l_new=l_new
        self.h_new=h_new
        self.num_annote = num_annote

        if mode == 'auto':
            data_array = np.array(data)
            self.l = np.min( data_array)
            self.h = np.max( data_array)
            # print self.l, self.h


    def map_range(self,v):
		return float(v-self.l)*((self.h_new-self.l_new)/float(self.h-self.l))+self.l_new

    def annote_it(self, y_gr  ):
        p = np.zeros( self.h - self.l + 1 )
        num_label = self.h - self.l + 1

        if num_label == 3:

            if y_gr == self.l or y_gr == self.h:
                p[ y_gr ] = .9
                if y_gr == self.h:
                    p[ y_gr - 1 ] = .1
                if y_gr == self.l:
                    p[ y_gr + 1 ] = .1
            else:
                p[ y_gr ] = .9
                p[ y_gr - 1 ]=.05
                p[ y_gr + 1 ]=.05
        else:
            if y_gr == self.l or y_gr == self.h:
                p[ y_gr ] = .8
                if y_gr == self.h:
                    p[ y_gr - 1 ] = .2
                if y_gr == self.l:
                    p[ y_gr + 1 ] = .2
            else:
                p[ y_gr ] = .8
                p[ y_gr - 1 ]=.1
                p[ y_gr + 1 ]=.1

        # print '-----'
        # print num_label
        # print p
        # print '---'
        return np.random.choice(range(num_label), self.num_annote, replace=True, p=p ) 

    # def annote_it_auto(self, y_gr):

    def annote_it_messidor(self,y_gr):
        p = np.zeros( self.h - self.l + 1 )
        num_label = self.h - self.l + 1
        if y_gr == self.l or y_gr == self.h:
            p[ y_gr ] = .9
            if y_gr == self.h:
                p[ y_gr - 1 ] = .1
            if y_gr == self.l:
                p[ y_gr + 1 ] = .1
        else:
            p[ y_gr ] = .9
            p[ y_gr - 1 ]=.05
            p[ y_gr + 1 ]=.05
        return np.random.choice(range(num_label), self.num_annote, replace=True, p=p ) 


    def get_error(self,y_h,y_ground):
        error = np.array([self.map_range(i) for i in (y_h-y_ground)])
        # print y_h
        # print error
        return np.mean(error**2)

class messidor_txt:
    def __init__(self,path,list_of_std):
        self.path = path
        self.list_of_std=list_of_std

    def process_data(self):
        data_file = self.path + 'messidor_features.arff'
        self.read_data( data_file)
        self.create_human_error()
        data_final = self.path + 'data'
        save( { 'x' : self.x, 'y': self.y, 'c': self.c } , data_final)

    def read_data(self,data_file):
        data = arff.loadarff(data_file)
        # print type(data)
        # print type(data[0])
        n= data[0].shape[0]
        d = len(data[0][0])-1
        # for d in data[0][0]:
        #     print '---'
        #     print d

        self.x=np.zeros((n,d))
        self.y=np.zeros(n)
        for row,row_ind in zip(data[0], range(n) ):
            for col,col_ind in zip( row , range(d+1) ) :
                if col_ind < d:
                    self.x[row_ind][col_ind] = float(col)
                else:
                    self.y[row_ind] = float(col)

    def create_human_error(self):
        self.c={}
        n=self.y.shape[0]
        for std in self.list_of_std:
            self.c[str(std)]= rand.normal(0,std,n)**2


            class Messidor:
    
                def __init__(self, path ):
            
                    self.path = path
                    self.annote= human_annotation( 0 , 2 , 0 , 1, 20 )
            
                    src_label = self.path + 'Annotation_Base11.csv'
                    label_file= self.path + 'labels_part'
                    # self.read_label_part(src_label, label_file)
                    # return
                    
                    src_dir= self.path + 'Base11/'
                    dest_dir= self.path + 'Images_part/' 
                    # self.read_images_part(src_dir, dest_dir, label_file)
                    
                    image_vector_file = self.path + 'out_Messidor'
                    data_file = self.path + 'data_part_Risk_edema' # Retino_grade'#'Risk_edema'
                    annote_file= self.path + 'annote_Risk_edema' # Retino_grade'
                    self.read_image_vector(image_vector_file, label_file ,  data_file, annote_file )
            
                def read_label_part(self,src_label,file_label):
            
                    
                    dict_label={'Retino_grade':{},'Risk_edema':{}}
                    rows = [] 
                    with open(src_label, 'r') as csvfile: 
                        csvreader = csv.reader(csvfile)
                        fields = csvreader.next() 
                        for row in csvreader: 
                            key=row[0][:-4]
                            dict_label['Retino_grade'][key] = int(row[2])
                            dict_label['Risk_edema'][key] = int(row[3])
                    save(dict_label,file_label)
                    # print max(dict_label['Retino_grade'].values())
                    # print max(dict_label['Risk_edema'].values())
            
                def read_images_part(self, src_dir, dest_dir, file_label):
                    
                    labels = load_data( file_label )
                    filenames = labels['Retino_grade'].keys()
                    for filename in filenames: 
                        file_path_src = src_dir + filename + '.tif'
                        file_path_dest  = dest_dir + filename + '.jpg'
                        image = Image.open(file_path_src)
                        image.save(file_path_dest)
            
                def read_image_vector(self, image_file, label_file, data_file, annote_file ):
                    image_vec_dict = load_data(image_file)
                    label_dict = load_data(label_file)
                    annote_dict = {}
                    x=[]
                    y=[]
                    c=[]
                    for name,vector in zip(image_vec_dict['names'], image_vec_dict['data']  ):
                        key = name.split('/')[-1][:-4]
                        x.append(vector)
                        y_ground = label_dict['Risk_edema'][key]#['Risk_edema']
                        y.append( self.annote.map_range( y_ground )  )
                        annotation = self.annote.annote_it_messidor( y_ground )
                        c.append( self.annote.get_error( annotation , y_ground ) )
                        annote_dict[ key ] = annotation
                        
                    save({'x':np.array(x), 'y':np.array(y), 'c':{'0.0':np.array(c)} }, data_file )
                    save( annote_dict, annote_file )
            

                    class Messidor:
    
                        def __init__(self, path ):
                    
                            self.path = path
                            self.annote= human_annotation( 0 , 2 , 0 , 1, 20 )
                    
                            src_label = self.path + 'Annotation_Base11.csv'
                            label_file= self.path + 'labels_part'
                            # self.read_label_part(src_label, label_file)
                            # return
                            
                            src_dir= self.path + 'Base11/'
                            dest_dir= self.path + 'Images_part/' 
                            # self.read_images_part(src_dir, dest_dir, label_file)
                            
                            image_vector_file = self.path + 'out_Messidor'
                            data_file = self.path + 'data_part_Risk_edema' # Retino_grade'#'Risk_edema'
                            annote_file= self.path + 'annote_Risk_edema' # Retino_grade'
                            self.read_image_vector(image_vector_file, label_file ,  data_file, annote_file )
                    
                        def read_label_part(self,src_label,file_label):
                    
                            
                            dict_label={'Retino_grade':{},'Risk_edema':{}}
                            rows = [] 
                            with open(src_label, 'r') as csvfile: 
                                csvreader = csv.reader(csvfile)
                                fields = csvreader.next() 
                                for row in csvreader: 
                                    key=row[0][:-4]
                                    dict_label['Retino_grade'][key] = int(row[2])
                                    dict_label['Risk_edema'][key] = int(row[3])
                            save(dict_label,file_label)
                            # print max(dict_label['Retino_grade'].values())
                            # print max(dict_label['Risk_edema'].values())
                    
                        def read_images_part(self, src_dir, dest_dir, file_label):
                            
                            labels = load_data( file_label )
                            filenames = labels['Retino_grade'].keys()
                            for filename in filenames: 
                                file_path_src = src_dir + filename + '.tif'
                                file_path_dest  = dest_dir + filename + '.jpg'
                                image = Image.open(file_path_src)
                                image.save(file_path_dest)
                    
                        def read_image_vector(self, image_file, label_file, data_file, annote_file ):
                            image_vec_dict = load_data(image_file)
                            label_dict = load_data(label_file)
                            annote_dict = {}
                            x=[]
                            y=[]
                            c=[]
                            for name,vector in zip(image_vec_dict['names'], image_vec_dict['data']  ):
                                key = name.split('/')[-1][:-4]
                                x.append(vector)
                                y_ground = label_dict['Risk_edema'][key]#['Risk_edema']
                                y.append( self.annote.map_range( y_ground )  )
                                annotation = self.annote.annote_it_messidor( y_ground )
                                c.append( self.annote.get_error( annotation , y_ground ) )
                                annote_dict[ key ] = annotation
                                
                            save({'x':np.array(x), 'y':np.array(y), 'c':{'0.0':np.array(c)} }, data_file )
                            save( annote_dict, annote_file )
                            class Messidor:
    
                                def __init__(self, path ):
                            
                                    self.path = path
                                    self.annote= human_annotation( 0 , 2 , 0 , 1, 20 )
                            
                                    src_label = self.path + 'Annotation_Base11.csv'
                                    label_file= self.path + 'labels_part'
                                    # self.read_label_part(src_label, label_file)
                                    # return
                                    
                                    src_dir= self.path + 'Base11/'
                                    dest_dir= self.path + 'Images_part/' 
                                    # self.read_images_part(src_dir, dest_dir, label_file)
                                    
                                    image_vector_file = self.path + 'out_Messidor'
                                    data_file = self.path + 'data_part_Risk_edema' # Retino_grade'#'Risk_edema'
                                    annote_file= self.path + 'annote_Risk_edema' # Retino_grade'
                                    self.read_image_vector(image_vector_file, label_file ,  data_file, annote_file )
                            
                                def read_label_part(self,src_label,file_label):
                            
                                    
                                    dict_label={'Retino_grade':{},'Risk_edema':{}}
                                    rows = [] 
                                    with open(src_label, 'r') as csvfile: 
                                        csvreader = csv.reader(csvfile)
                                        fields = csvreader.next() 
                                        for row in csvreader: 
                                            key=row[0][:-4]
                                            dict_label['Retino_grade'][key] = int(row[2])
                                            dict_label['Risk_edema'][key] = int(row[3])
                                    save(dict_label,file_label)
                                    # print max(dict_label['Retino_grade'].values())
                                    # print max(dict_label['Risk_edema'].values())
                            
                                def read_images_part(self, src_dir, dest_dir, file_label):
                                    
                                    labels = load_data( file_label )
                                    filenames = labels['Retino_grade'].keys()
                                    for filename in filenames: 
                                        file_path_src = src_dir + filename + '.tif'
                                        file_path_dest  = dest_dir + filename + '.jpg'
                                        image = Image.open(file_path_src)
                                        image.save(file_path_dest)
                            
                                def read_image_vector(self, image_file, label_file, data_file, annote_file ):
                                    image_vec_dict = load_data(image_file)
                                    label_dict = load_data(label_file)
                                    annote_dict = {}
                                    x=[]
                                    y=[]
                                    c=[]
                                    for name,vector in zip(image_vec_dict['names'], image_vec_dict['data']  ):
                                        key = name.split('/')[-1][:-4]
                                        x.append(vector)
                                        y_ground = label_dict['Risk_edema'][key]#['Risk_edema']
                                        y.append( self.annote.map_range( y_ground )  )
                                        annotation = self.annote.annote_it_messidor( y_ground )
                                        c.append( self.annote.get_error( annotation , y_ground ) )
                                        annote_dict[ key ] = annotation
                                        
                                    save({'x':np.array(x), 'y':np.array(y), 'c':{'0.0':np.array(c)} }, data_file )
                                    save( annote_dict, annote_file )
                                                                                
class STARE:

    def __init__(self, path ):

        self.path = path
        src_label = self.path + 'annotations/'
        label_file= self.path + 'labels'
        # self.read_label_part(src_label, label_file)
        # return
        
        src_dir= self.path + 'all-images/'
        dest_dir= self.path + 'Images/' 
        label_file= self.path + 'labels'
        # self.read_images_part(src_dir, dest_dir, label_file)
        
        selected_labels = [ 5,11 ]
        for i in selected_labels:
            image_vector_file = self.path + 'out_STARE'
            data_file = self.path + str(i) + '/data' 
            annote_file= self.path + str(i) + '/annote' 
            # self.read_image_vector(image_vector_file, label_file ,  data_file, annote_file, i )

            data_file_split = self.path+ str(i) + '/data_split'
            # split_data(0.8, data_file, data_file_split)

            data_file_pca = data_file_split+ '_pca'
            # process_data(data_file_split, data_file_pca, num_features=100 )
            data_file_txt = '../Real_Data/text_files/'+ self.path.split('/')[2]+'_'+str(i)
            write_file_to_txt(  data_file_pca, data_file_txt, annote_file, image_vector_file )

    def read_label_part(self,src_label,file_label):
        dict_label={ } 
        for f in listdir(src_label):
            name = f.split('.')[0]
            with open( src_label + f, 'r') as file: 
                line = file.readline()
                dict_label[name] = [ int(l) for l in line.strip() ]
        save(dict_label,file_label)
        
    def read_images_part(self, src_dir, dest_dir, file_label):
        labels = load_data( file_label )
        for filename in labels['names']: 
            file_path_src = src_dir + filename + '.ppm'
            file_path_dest  = dest_dir + filename + '.jpg'
            image = Image.open(file_path_src)
            image.save(file_path_dest)
   
    def read_image_vector(self, image_file, label_file, data_file, annote_file , i ):
        
        image_vec_dict = load_data(image_file)
        label_dict = load_data(label_file)
        
        y_unscaled=[ label_dict[ key ][i] for key in label_dict.keys()]
        self.annote= human_annotation( 0 , 0 , 0 , 1, 20 , mode = 'auto', data=y_unscaled )
                
        x=[]
        y=[]
        c=[]
        annote_dict = {}
        for name,vector in zip(image_vec_dict['names'], image_vec_dict['data']  ):
            key = name.split('/')[-1][:-4]
            x.append(vector)
            y_ground = label_dict[key][i]
            y.append( self.annote.map_range( y_ground )  )
            annotation = self.annote.annote_it( y_ground )
            c.append( self.annote.get_error( annotation , y_ground ) )
            annote_dict[ key ] = annotation
                       
        save({'x':np.array(x), 'y':np.array(y), 'c':{'0.0':np.array(c)} }, data_file )
        save( annote_dict, annote_file )
        
class Messidor:
    
    def __init__(self, path ):

        self.path = path
        self.annote= human_annotation( 0 , 2 , 0 , 1, 20 )

        src_label = self.path + 'Annotation_Base11.csv'
        label_file= self.path + 'labels_part'
        # self.read_label_part(src_label, label_file)
        # return
        
        src_dir= self.path + 'Base11/'
        dest_dir= self.path + 'Images_part/' 
        # self.read_images_part(src_dir, dest_dir, label_file)
        
        choice = 'Risk_edema'
        image_vector_file = self.path + 'out_Messidor'
        data_file = self.path + 'data_part_' + choice 
        annote_file= self.path + 'annote_' + choice
        # self.read_image_vector(image_vector_file, label_file ,  data_file, annote_file )

        data_file_split = self.path+ choice + '/data_split'
        # split_data(0.8, data_file, data_file_split)

        data_file_pca = data_file_split  + '_pca'
        # process_data(data_file_split, data_file_pca, num_features=100 )
        data_file_txt = '../Real_Data/text_files/'+ self.path.split('/')[2]+'_'+choice
        write_file_to_txt(  data_file_pca, data_file_txt, annote_file, image_vector_file )


    def read_label_part(self,src_label,file_label):

        
        dict_label={'Retino_grade':{},'Risk_edema':{}}
        rows = [] 
        with open(src_label, 'r') as csvfile: 
            csvreader = csv.reader(csvfile)
            fields = csvreader.next() 
            for row in csvreader: 
                key=row[0][:-4]
                dict_label['Retino_grade'][key] = int(row[2])
                dict_label['Risk_edema'][key] = int(row[3])
        save(dict_label,file_label)
        # print max(dict_label['Retino_grade'].values())
        # print max(dict_label['Risk_edema'].values())

    def read_images_part(self, src_dir, dest_dir, file_label):
        
        labels = load_data( file_label )
        filenames = labels['Retino_grade'].keys()
        for filename in filenames: 
            file_path_src = src_dir + filename + '.tif'
            file_path_dest  = dest_dir + filename + '.jpg'
            image = Image.open(file_path_src)
            image.save(file_path_dest)

    def read_image_vector(self, image_file, label_file, data_file, annote_file ):
        image_vec_dict = load_data(image_file)
        label_dict = load_data(label_file)
        annote_dict = {}
        x=[]
        y=[]
        c=[]
        for name,vector in zip(image_vec_dict['names'], image_vec_dict['data']  ):
            key = name.split('/')[-1][:-4]
            x.append(vector)
            y_ground = label_dict['Risk_edema'][key]#['Risk_edema']
            y.append( self.annote.map_range( y_ground )  )
            annotation = self.annote.annote_it_messidor( y_ground )
            c.append( self.annote.get_error( annotation , y_ground ) )
            annote_dict[ key ] = annotation
            
        save({'x':np.array(x), 'y':np.array(y), 'c':{'0.0':np.array(c)} }, data_file )
        save( annote_dict, annote_file )

class EyePAC:
    
    def __init__(self, path ):

        self.path = path
        self.annote= human_annotation( 0 , 4 , 0 , 1, 20 )

        src_label = self.path + 'trainLabels.csv'
        label_file= self.path + 'labels_part'
        # self.read_label_part(src_label, label_file)
        # return
        
        src_dir= self.path + 'Images_Random/'
        dest_dir= self.path + 'Images_part/' 
        # self.read_images_part(src_dir, dest_dir, file_label)
        
        image_vector_file = self.path + 'out'
        data_file = self.path + 'data_part'
        annote_file= self.path + 'annote'
        # self.read_image_vector(image_vector_file, label_file ,  data_file, annote_file )
        
        data_file_split = self.path + 'data_split'
        # split_data(0.8, data_file, data_file_split)

        data_file_pca = data_file_split + '_pca'
        # process_data(data_file_split, data_file_pca, num_features=100 )
        data_file_txt = '../Real_Data/text_files/'+ self.path.split('/')[2]
        write_file_to_txt(  data_file_pca, data_file_txt, annote_file, image_vector_file )


    def read_label_part(self,src_label,file_label):
        
        dict_label={}
        rows = [] 
        with open(src_label, 'r') as csvfile: 
            csvreader = csv.reader(csvfile)
            fields = csvreader.next() 
            ind=0
            for row in csvreader: 
                dict_label[row[0]] = int(row[1])
                ind+=1
                if ind==1000:
                    # print dict_label
                    save(dict_label,file_label)

    def read_images_part(self, src_dir, dest_dir, file_label):
        
        labels = load_data( file_label )
        filenames = labels.keys()
        for filename in filenames: 
            file_path_src = src_dir + filename + '.jpeg'
            file_path_dest  = dest_dir + filename + '.jpg'
            # print file_path_src
            # print file_path_dest
            image = Image.open(file_path_src)
            image.save(file_path_dest)
            # shutil.copy( file_path_src, file_path_dest )
            # return

    def read_image_vector(self, image_file, label_file, data_file, annote_file ):
        image_vec_dict = load_data(image_file)
        label_dict = load_data(label_file)
        annote_dict = {}
        x=[]
        y=[]
        c=[]
        ind =0
        for name,vector in zip(image_vec_dict['names'], image_vec_dict['data']  ):
            key = name.split('/')[-1][:-4]
            x.append(vector)
            y_ground = label_dict[key]
            y.append( self.annote.map_range( y_ground )  )
            annotation = self.annote.annote_it( y_ground )
            c.append( self.annote.get_error( annotation , y_ground ) )
            annote_dict[ key ] = annotation
            ind+=1
            if ind == 750:
                break
        save({'x':np.array(x), 'y':np.array(y), 'c':{'0.0':np.array(c)} }, data_file )
        save( annote_dict, annote_file )

def main():
    # list_of_std = [0.001,0.01,0.1]
    # path = '../Real_Data/CheXpert/'
    # obj = CheXpert( path ) 

    # path = '../Real_Data/STARE/'
    # obj=STARE( path )

    # path = '../Real_Data/Messidor/Messidor_txt/'
    # obj=messidor_txt( path , list_of_std )
    # obj.process_data()

    # path = '../Real_Data/EyePAC/'
    # obj=EyePAC( path  )
    
    path = '../Real_Data/Messidor/MESSIDOR/'
    obj = Messidor( path )

    

if __name__=="__main__":
	main()

