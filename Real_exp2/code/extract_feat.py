import numpy as np
import pickle
import h5py
def save(obj,output_file):
	with open(output_file+'.pkl' , 'wb') as f:
		pickle.dump( obj , f , pickle.HIGHEST_PROTOCOL)
filename = 'features.h5'
f = h5py.File(filename, 'r')
# f = h5py.File(file_name, mode)
# Studying the structure of the file by printing what HDF5 groups are present

# for key in f.keys():
#     print(key) #Names of the groups in HDF5 file.
# Extracting the data

# #Get the HDF5 group
# print type( f['filenames'].value)
names =  f['filenames'].value
data = f['resnet_v2_101']['logits'].value
data_final = np.squeeze(data)
data_dict = {'names':names, 'data':data_final}
save(data_dict,'out')
f.close()
