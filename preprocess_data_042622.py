import numpy as np
import matplotlib.pyplot as plt
import torch
from os import listdir
from os.path import isfile, join
import gc

mypath = '/gpfs/group/vav5084/default/ashley/knemulator/kn_sim_cube_v1/'
onlyfiles = [f for f in listdir(mypath) if (('spec' in f) & isfile(join(mypath, f)))]
#param stuff
all_data = np.zeros((len(onlyfiles),73,1024,56))
all_params = np.zeros((len(onlyfiles),4))
for i,f in enumerate(onlyfiles):
	print(i,f,len(onlyfiles))
	md = float(f.split('md')[1].split('_')[0])
	vd = float(f.split('vd')[1].split('_')[0])
	mw = float(f.split('mw')[1].split('_')[0])
	vw = float(f.split('vw')[1].split('_')[0])
	mdstr = f.split('md')[1].split('_')[0]
	vdstr = f.split('vd')[1].split('_')[0]
	mwstr =f.split('mw')[1].split('_')[0]
	vwstr =  f.split('vw')[1].split('_')[0]
	data = np.loadtxt(mypath+f)
	data = data.reshape(-1,1024,56)
	for t in np.arange(len(data),dtype=int):
		for angle in np.arange(np.shape(data)[-1],dtype=int):
			fname = 'kn'+mdstr+'-'+vdstr+'-'+mwstr+'-'+vwstr+'-'+str(vd)+'-'+str(t)+'-'+str(angle)+'.pt'
			x = torch.tensor(data[t,:,angle])
			torch.save(x,'./pt_files/'+fname)
	gc.collect()
