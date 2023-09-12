# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import sys
    sys.path.append('../..')

import os
import numpy as np
import matplotlib.pyplot as plt
from utils.picasso_hdf5 import save as save_hdf5

import h5py

#def save(fn, xyIBg, crlb, framenum, imgshape, sigmaX, sigmaY, extraColumns=None):   

import scipy.io as sio

#def save_points(fn, obj):
	

#save_hdf5()

#fn = 'C:/dev/simflux/data/06082019/object_filamentousWLC_05082019_rho1E3.mat'
fn= 'C:/dev/simflux/data/rasmus-sim/object_filamentousWLC_20192407_rho20E3.mat'

with h5py.File(fn, 'r') as f:
	print(f.keys())

	obj=f['object']
	d = np.array(obj).T
	
	framenum = d[:,3]
	xyIBg = np.zeros((len(d),4))
	xyIBg[:,[0,1,2]] = d[:,[0,1,4]]
	xyIBg[:,3] = 20
	sigmaX = 1.665
	sigmaY = 1.665
	
	# Just some semi reasonable numbers
	crlb = np.zeros((len(d),4))
	crlb[:,[0,1]] = 0.15
	crlb[:,2] = 0.1
	crlb[:,3] = np.sqrt(np.mean(xyIBg[:,2]))
	imgshape = np.ceil(np.max(d[:,[0,1]],0)).astype(int)
	outfn = os.path.splitext(fn)[0]+"_ground_truth.hdf5"
	save_hdf5(outfn,xyIBg, crlb, framenum, imgshape, sigmaX, sigmaY)
	