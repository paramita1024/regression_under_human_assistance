import numpy.random as rand
import codecs
import csv 
import random
import fasttext
from myutil import *
import numpy as np
import numpy.linalg as LA

class preprocess_triage_real_data:
	def __init__(self):
		pass

	def preprocess_movielens_data(self,path):
		rating_src = path + 'ratings.csv'
		rating_file =path + 'ratings_dict'
		# self.read_rating_file(rating_src,rating_file)
		
		genome_src=path+'genome-scores.csv'
		genome_file = path + 'genome-score'
		# self.read_genome_score(genome_src, genome_file)
		# mid 131170
		# tid 1128

		# self.get_subset_full_data(path)
		src = path + 'data_tr'
		dest = path + 'data_tr_mat'
		# self.format_data(src,dest)

		src = path + 'data_tr_mat'
		dest = path + 'data_tr_splitted'
		# self.split_data(0.8, src, dest)
	
	def format_data(self,src,dest):
		data_src=load_data(src)
		# data = {'X':{}, 'Y_full':{}, 'indices':sel_indices }
		
		n_data =  len(data_src['indices'])
		n_feature = 1128
		data={}
		data['x']=np.zeros((n_data,n_feature))
		data['y'] = np.zeros(n_data)
		data['c'] = np.zeros(n_data)
		for ind_new,ind_old in zip(range(n_data), data_src['indices'] ):
			data['x'][ind_new] = data_src['X'][ind_old][1:n_feature+1]
			human_rating = np.array([ self.map_range(i,0,5,0,1) for i in data_src['Y_full'][ind_old] ])
			data['y'][ind_new] = np.average(human_rating)
			data['c'][ind_new] = np.average((human_rating - data['y'][ind_new])**2 )
		save(data,dest)

	def get_subset_full_data(self,path):
		
		gnome_file = path + 'genome-score'
		g=load_data(gnome_file)
		mids_g = g.keys()
		# del g 

		rating_file = path + 'ratings_dict'
		r=load_data(rating_file)
		mids_r = r.keys()
		# del r

		g_and_r = [ i for i in  mids_g if i in mids_r ]
		sel_indices = g_and_r[:1500]
		data = {'X':{}, 'Y_full':{}, 'indices':sel_indices }
		for i in sel_indices:
			data['X'][i] = g[i]
			data['Y_full'][i] = r[i] 
		data_file = path + 't1'#'data_tr'
		save(data,data_file)

	def read_genome_score(self,src,dest):
		scores={}
		max_tags=2000
		max_mid= 0 
		max_tid = 0 
		with open(src,'r') as f:
			header = f.readline()
			l = f.readline()
			while l :
				mid,tid,score = l.split(',')
				if int(mid) > max_mid :
					max_mid = int(mid)
				if int(tid ) > max_tid :
					max_tid = int(tid)
				if not mid in scores:
					scores[mid]= np.zeros(max_tags)
				scores[mid][int(tid)] = float(score)
				l = f.readline()
		# for k in tags.keys():
		# 	print tags[k]
		print 'mid',max_mid
		print 'tid',max_tid
		save(scores,dest)

	def read_genome_desc(self,src,dest):
		# ind = 0 
		tags={}
		num_discarded = 0
		ind=0
		with open(src,'r') as f:
			header = f.readline()
			l = f.readline()
			while l :
				print l
				# words=l.split(',')
				# if len(words) == 4:
				# 	uid,mid,tag,times = l.split(',')
				# 	if mid in tags:
				# 		tags[mid].append(tag)
				# 	else:
				# 		tags[mid]=[tag]
				# else:
				# 	if '"' in l:
				# 		mid = l.split('"',3)[0].split(',')[1]
				# 		tag=l.split('"',3)[1]
				# 		if mid in tags:
				# 			tags[mid].append(tag)
				# 		else:
				# 			tags[mid]=[tag]
				# 	else:
				# 		num_discarded +=1 
				# 		print l
				l = f.readline()
				# ind+=1
		# print num_discarded
		# print ind
		# for k in tags.keys():
		# 	print tags[k]
		save(tags,dest)

	def read_tag_file(self,src_file,tag_file):
		# ind = 0 
		tags={}
		num_discarded = 0
		ind=0
		with open(src_file,'r') as f:
			header = f.readline()
			# print header
			l = f.readline()
			
			while l :
				# print l
				# print l.split()
				words=l.split(',')
				if len(words) == 4:
					uid,mid,tag,times = l.split(',')
					if mid in tags:
						tags[mid].append(tag)
					else:
						tags[mid]=[tag]
				else:
					if '"' in l:
						mid = l.split('"',3)[0].split(',')[1]
						tag=l.split('"',3)[1]
						if mid in tags:
							tags[mid].append(tag)
						else:
							tags[mid]=[tag]
					else:
						num_discarded +=1 
						print l
				l = f.readline()
				ind+=1
		# print num_discarded
		# print ind
		for k in tags.keys():
			print tags[k]
		save(tag,tag_file)
				
	def read_rating_file(self,src_file,rating_file):
		rating={}
		with open(src_file,'r') as f:
			header = f.readline()
			l = f.readline()
			while l :
				uid,mid,score,times = l.split(',')
				if mid in rating:
					rating[mid].append(float(score) )
				else:
					rating[mid]=[ float(score)  ]
				l = f.readline()
		save(rating,rating_file)
				
	def process_hate_speech_data(self,src_file,dest_file):
		# ind = 0 
		with open(src_file,'r') as f:
			f.readline()
			dict_tweet={}
			response_list=[]
			human_annotation_list=[]
			while True:
				line_full =  f.readline()
				print '&&',line_full,'&&'
				if not line_full:
					save({'tweets':dict_tweet,'y':response_list,'y_h':human_annotation_list},dest_file)
					return 
					# return dict_tweet,response_list,human_annotation_list
				else:
					if line_full.isspace():
						print 'empty'
					else:
						line=line_full.split(',',7)
						if len(line) == 7:
							tid=line[0]
							tweet=line[-1]
							dict_tweet[tid]=tweet
							y,y_h=self.get_annotations(line[1:-1])
							response_list.append(y)
							human_annotation_list.append(y_h)

	def get_annotations(self,list_of_arg):
		human_response  = []
		for i in [1,2,3]:
			if int(list_of_arg[i]) > 0 :
				human_response.extend([i-1]*int(list_of_arg[i]))
		response = int(list_of_arg[-1])
		return response,human_response

	def dict_to_txt(self,tweet_dict,file_w):
		with open(file_w,'w') as f:
			for tweet in tweet_dict.values():
				f.write(tweet)

	def map_range(self,v,l,h,l_new,h_new):
		return float(v-l)*((h_new-l_new)/float(h-l))+l_new

	def convert_tweet_to_vector(self,file_dict,file_vec,file_tweet):
		data_dict=load_data(file_dict)
		data_vec={}
		n_data=len(data_dict['y'])
		data_vec['y']=np.array([ self.map_range(i,0,2,0,1) for i in data_dict['y'] ])
		data_vec['c']=np.zeros(n_data)
		for ind,human_pred,response in zip(range(n_data),data_dict['y_h'],data_vec['y']):
			human_pred_scaled = [self.map_range(i,0,2,0,1) for i in human_pred ]
			data_vec['c'][ind] = (np.mean( np.array( human_pred_scaled ) ) - float(response) )**2
		# self.dict_to_txt(data_dict['tweets'],tweet_file)
		model = fasttext.train_unsupervised(file_tweet, model='skipgram')
		x=[]
		for tid in data_dict['tweets'].keys():
			tweet = data_dict['tweets'][tid].replace('\n',' ')
			x.append(model.get_sentence_vector(tweet).flatten() )
		data_vec['x']=np.array(x)
		# or, cbow model :
		# model = fasttext.train_unsupervised('data.txt', model='cbow')
		save(data_vec,file_vec)

	def truncate_data(self,data_file,data_file_tr):
		data=load_data(data_file)
		n=data['y'].shape[0]
		n_tr=int(n/4)
		data['x']=data['x'][:n_tr]
		data['y']=data['y'][:n_tr]
		data['c']=data['c'][:n_tr]
		save(data,data_file_tr)

	def split_data(self,frac,file_data,file_data_split):
		
		data = load_data(file_data)

		print 'x' , data['x'].shape 
		print 'y' , data['y'].shape
		print 'c' , data['c'].shape
		# return 
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
		data_split['c']=data['c'][indices_train]
		test={}
		test['X']=data['x'][indices_test]
		test['Y']=data['y'][indices_test]
		test['c']=data['c'][indices_test]
		data_split['test']=test
		data_split['dist_mat']=np.zeros((num_test,num_train))
		for te in range(num_test):
			for tr in range(num_train):
				data_split['dist_mat'][te,tr]=LA.norm( test['X'][te] - data_split['X'][tr] )
		save( data_split, file_data_split)

	def preprocess_BRAND_DATA(self,path):
		
		src_ann= path + 'mturk_ori_annotate_first.csv'
		src_txt= path + 'mturk_ori_seqno_text_mapping.csv'
		data_x= path + 'tmp/data_x'
		data_y= path + 'tmp/data_y'
		txt_file = path + 'tmp/text'
		data_x_vec = path + 'tmp/data_x_vec'
		data_part = path + 'data'
		
		# self.read_text_BRAND(src_txt,data_x)
		# self.process_text_BRAND(data_x,txt_file,data_x_vec)
		# self.read_label_BRAND(src_ann, data_y)

		# self.split_labels_BRAND(data_x_vec,data_y,data_part)
		for i in range(5):
			self.split_data(0.8,data_part+str(i+1)+'/data',data_part+str(i+1)+'/data_split')
		# data_vec_split_file = path + 'data_ht4_vec_split_1'
		# self.split_data(0.8,data_vec_file,data_vec_split_file)
	
	def process_text_BRAND(self,src,txt_file,dest):
		data_dict=load_data(src)
		self.dict_to_txt(data_dict,txt_file)
		model = fasttext.train_unsupervised(txt_file, model='skipgram')
		x={}
		for tid in data_dict.keys():
			article = data_dict[tid].replace('\n',' ').decode('utf-8')
			x[tid]=model.get_sentence_vector(article).flatten() 
		save(x,dest)
		
	def read_text_BRAND(self,src,dest):
		dict_data={}
		rows = [] 
		with open(src, 'r') as csvfile: 
			csvreader = csv.reader(csvfile)
			fields = csvreader.next() 
			for row in csvreader: 
				rows.append(row) 
		for row in rows: 
			dict_data[row[1]]=row[2]
		# print dict_data['0']
		save(dict_data,dest)

	def read_label_BRAND(self,src,dest):
		data_dict={}
		rows = [] 
		with open(src, 'r') as csvfile: 
			csvreader = csv.reader(csvfile)
			fields = csvreader.next() 
			for row in csvreader: 
				rows.append(row) 
		# print('Field names are:' + ', '.join(field for field in fields)) 
		for row in rows: 
			# print row
			# return 	
			l = []
			for s in row[1:6]:
				if s.isdigit():
					l.append(int(s))
			# print l
			if row[0] not in  data_dict:
				data_dict[row[0]] = []
			data_dict[row[0]].append( l )
		save( data_dict, dest )

	def split_labels_BRAND(self,src_x,src_y,dest): 

		data_y=load_data(src_y)
		data_x=load_data(src_x)
		for i in range(5):
			x=[]
			y=[]
			c=[]
			for id in data_x.keys():
				if id in data_y:
					tmp = np.array(data_y[id])
					if len(tmp.shape) > 1 :
						x.append(data_x[id])
						y_h = np.array([ self.map_range(j,1,5,0,1) for j in tmp[:,i].flatten() ])
						y.append(np.mean(y_h))
						c.append(np.mean( (y_h - np.mean(y_h))**2 ))
			save({'x':np.array(x),'y':np.array(y),'c':np.array(c)}, dest+str(i+1)+'/data')

	def preprocess_BRAND_DATA_BIN(self,path):
		# src_file = path + 'heldout_essays_16thMay.csv'
		# dest_file = path + 'data_heldout'

		src_file = path + 'ht4_essays_data_12thFeb.csv'
		data_file = path + 'data_ht4'
		# self.read_csv_file(src_file,data_file)

		
		blog_file = path + 'blog_file'
		data_vec_file = path + 'data_ht4_vec_1'
		# self.format_data_BRAND(blog_file,data_file,data_vec_file)

		data_vec_split_file = path + 'data_ht4_vec_split_1'
		# self.split_data(0.8,data_vec_file,data_vec_split_file)

	def format_data_BRAND(self,blog_file,data_file,data_vec_file):
		
		data = load_data(data_file)
		n=data['y_h'].shape[0]
		# print n
		# return 
		# only_txt = {i:data[i]['txt'] for i in data['data'].keys()}
		# self.dict_to_txt(only_txt, blog_file)		
		model = fasttext.train_unsupervised(blog_file, model='skipgram')
		data_vec={'y':np.zeros(n),'c':np.zeros(n)}
		x=[]
		for tid,y_h in zip(data['data'].keys(),data['y_h']):
			blog = data['data'][tid]['txt'].replace('\n',' ').decode('utf-8')
			print blog
			print '****************************************************'
			x.append(model.get_sentence_vector(blog).flatten() )
			# print y_h
			data_vec['y'][int(tid)] = np.mean(y_h)
			data_vec['c'][int(tid)] = np.mean((y_h-np.mean(y_h))**2)*0.01
		# return 
		plt.plot(data_vec['y'],label='y')
		plt.plot(data_vec['c'],label='c')
		plt.legend()
		plt.show()
		data_vec['x']=np.array(x)

		save(data_vec, data_vec_file)

	def read_csv_file(self,src_file,dest_file):
		with open(src_file,'r') as f:
			l = f.readline()
			# print l 
			ind =0 
			dict_data = {}
			rest_var_list=[]
			l = f.readline()
			while l :
				if l.strip():
					auth_str = l.split(',')[0].replace('"','')
					rest_id = l.split(',')[-5:]
					text = ','.join(l.split(',')[1:-5])
					if text.strip():
						# print '---------'
						# print auth_str
						# print '*****'
						# print text
						
						dict_data[str(ind)] = {}
						dict_data[str(ind)]['txt']=text
						dict_data[str(ind)]['auth_id']=auth_str
						# print rest_id
						y_l = [1 if s=='"y"' else 0 for s in rest_id ]
						rest_var_list.append(y_l)
						print y_l
						ind += 1 
						# print '----',ind,'-----'
				l = f.readline()
		save({'data':dict_data,'y_h':np.array(rest_var_list)}, dest_file)

	def preprocess_VLOG_data(self,path):
		src = path + 'YoutubeVlogAnnotatedDataset.csv'
		dest = path + 'data_dict'
		self.read_csv_data_VLOG(src,dest)
	
	def read_csv_data_VLOG(self,src,dest):
		fields = [] 
		rows = [] 
		with open(src, 'r') as csvfile: 
			csvreader = csv.reader(csvfile)
			fields = csvreader.next() 
			for row in csvreader: 
				rows.append(row) 
			# print("Total no. of rows: %d"%(csvreader.line_num)) 
		# print('Field names are:' + ', '.join(field for field in fields)) 
		# print len(fields)
		dict_data = {}
		num_del = 0 
		for row,ind in zip(rows,range(len(rows))): 
			# for col in row: 
			# 	print("%10s"%col), 
			num_empty=0
			dict_data[str(ind)]={'ImageTags':{}}
			for col,field_name in zip(row,fields):
				if 'ImageTag' in field_name:
					if 'conf' not in field_name:
						tmp = col
					else:
						# print '***'+tmp+'***'
						# print '***'+col+'***'
						if tmp.strip() and col.strip():
							dict_data[str(ind)]['ImageTags'][tmp]=float(col)
						else:
							num_empty +=1
				else:
					dict_data[str(ind)][field_name]=col
			if num_empty > 30:
				del dict_data[str(ind)]
				num_del +=1
		save({'data':dict_data, 'fields':fields},dest)
		for key in dict_data.keys():
			print '-------------------------------------'
			print dict_data[key]
			

def main():
	# path='../Real_Data/BRAND_DATA/'  
	# obj=preprocess_triage_real_data()
	# obj.preprocess_BRAND_DATA(path)
	#---------------------------
	# path='../Real_Data/VLOG/' 
	# obj=preprocess_triage_real_data()
	# obj.preprocess_VLOG_data(path)
	#---------------------------------
	# path='../Real_Data/BRAND_DATA_BIN/'  
	# obj=preprocess_triage_real_data()
	# obj.preprocess_BRAND_DATA_BIN(path)
	#---------------------------------
	# path='../Real_Data/Movielens/ml-20m/'
	# r =load_data(path + 'ratings_dict')
	# obj=preprocess_triage_real_data()
	# obj.preprocess_movielens_data(path)
	#--------------
	path='../Real_Data/Hatespeech/Davidson/'
	src_file = path+'labeled_data.csv'
	obj=preprocess_triage_real_data()
	dest_file = path + 'data'
	tweet_file = path + 'tweets.txt'
	vec_file = path + 'data_vectorized'
	vec_tr_file= path + 'data_vectorized_tr'
	vec_split_file = path + 'input_tr'

	# obj.process_hate_speech_data(src_file,dest_file)
	# obj.convert_tweet_to_vector(dest_file,vec_file,tweet_file)
	# obj.truncate_data(vec_file, vec_tr_file)
	# obj.split_data(0.8, vec_tr_file , vec_split_file)


if __name__=="__main__":
	main()


# tag_src = path + 'tags.csv'
		# tag_file = path + 'tag'
		# # self.read_tag_file(tag_src,tag_file)
		# genome_src=path+'genome-tags.csv'
		# genome_file = path + 'genome-tags'
		# # self.read_genome_desc(genome_src, genome_file)
