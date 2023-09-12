# -*- coding: utf-8 -*-
import h5py 
import yaml
import numpy as np
import os


def load(fn):
    with h5py.File(fn, 'r') as f:
        locs = f['locs']
       
        estim = np.zeros((len(locs),4))
        estim[:,0] = locs['x']
        estim[:,1] = locs['y']
        estim[:,2] = locs['photons']
        estim[:,3] = locs['bg']
        
        crlb = np.zeros((len(locs),4))
        crlb[:,0] = locs['lpx']
        crlb[:,1] = locs['lpy']
        crlb[:,2] = locs['lI']
        crlb[:,3] = locs['lbg']
        
        info_fn = os.path.splitext(fn)[0] + ".yaml" 
        with open(info_fn, "r") as file:
            if hasattr(yaml, 'unsafe_load'):
                obj = yaml.unsafe_load_all(file)
            else:
                obj = yaml.load_all(file)
            obj=list(obj)[0]
            imgshape=np.array([obj['Height'],obj['Width']])
            
        sx = np.array(locs['sx'])
        sy = np.array(locs['sy'])
        
        return estim,locs['frame'],crlb, imgshape,sx,sy
        
        
def save(fn, coords, crlb, framenum, imgshape, sigmaX, sigmaY, extraColumns=None):   
    print(f"Saving hdf5 to {fn}")
    
    is3D = coords.shape[1] == 5
    with h5py.File(fn, 'w') as f:
        dtype = [('frame', '<u4'), 
                 ('x', '<f4'), ('y', '<f4'),
                 ('photons', '<f4'), 
                 ('sx', '<f4'), ('sy', '<f4'), 
                 ('bg', '<f4'), 
                 ('lpx', '<f4'), ('lpy', '<f4'), 
                 ('lI', '<f4'), ('lbg', '<f4'), 
                 ('ellipticity', '<f4'), 
                 ('net_gradient', '<f4')]
        
        if is3D:
            for fld in [('z', '<f4'), ('lpz', '<f4')]:
                dtype.append(fld)
        
        if extraColumns is not None:
            for k in extraColumns.keys():
                dtype.append((k,extraColumns[k].dtype,extraColumns[k].shape[1:]))
        
        locs = f.create_dataset('locs', shape=(len(coords),), dtype=dtype)
        locs['frame'] = framenum
        locs['x'] = coords[:,0]
        locs['y'] = coords[:,1]
        
        Nidx=2
        bgIdx=3
        if is3D:
            Nidx=3
            bgIdx=4
            locs['z'] = coords[:,2]
            locs['lpz'] = crlb[:,3]
        
        locs['photons'] = coords[:,Nidx]
        locs['bg'] = coords[:,bgIdx]
        locs['sx'] = sigmaX
        locs['sy'] = sigmaY
        locs['lpx'] = crlb[:,0]
        locs['lpy'] = crlb[:,1]
        locs['lI'] = crlb[:,2],
        locs['lbg'] = crlb[:,3]
        locs['net_gradient'] = 0
        
        if extraColumns is not None:
            for k in extraColumns.keys():
                locs[k] = extraColumns[k] 
                
        info =  {'Byte Order': '<',
                 'Camera': 'Dont know' ,
                 'Data Type': 'uint16',
                 'File': fn,
                 'Frames': int(np.max(framenum)+1 if len(framenum)>0 else 0),
                 'Width': int(imgshape[1]),
                 'Height': int(imgshape[0])
                 }
        
        info_fn = os.path.splitext(fn)[0] + ".yaml" 
        with open(info_fn, "w") as file:
            yaml.dump(info, file, default_flow_style=False) 

    return fn