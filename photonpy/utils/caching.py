# -*- coding: utf-8 -*-
import os
import pickle
import tqdm
import numpy as np



    
def equal_cache_cfg(output_fn, cfg, input_fn):
    """
    Returns true if the config file associated with data file data_fn contains the same value as cfg
    """ 
    mtime = os.path.getmtime(input_fn)
    cfg_fn = os.path.splitext(output_fn)[0]+"_cfg.pickle"
    if not os.path.exists(cfg_fn):
        return False
    with open(cfg_fn,"rb") as f:
        d = pickle.load(f)
        if len(d) != 2:
            return False
        
        stored, stored_mtime = d
        
        if stored_mtime != mtime:
            return False
        
        try:
            # Note that we can't just do 'return stored == cfg'. 
            # If one of the values in a dictionary is a numpy array, 
            # we will get "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"
            # See https://stackoverflow.com/questions/26420911/comparing-two-dictionaries-with-numpy-matrices-as-values
            np.testing.assert_equal(stored, cfg)
        except:
            return False

        return True
            
def load_cache_cfg(output_fn):
    cfg_fn = os.path.splitext(output_fn)[0]+"_cfg.pickle"
    with open(cfg_fn,"rb") as f:
        d = pickle.load(f)
        return d[0]
    
    
def save_cache_cfg(output_fn, cfg, input_fn):
    mtime = os.path.getmtime(input_fn)
    cfg_fn = os.path.splitext(output_fn)[0]+"_cfg.pickle"
    with open(cfg_fn,"wb") as f:
        pickle.dump((cfg, mtime),f)
    

# Call func(path, cfg) only if the cached version has a different value of 'cfg'.
# If equal, return the cached data
def read(cache_fn, cfg, func, rebuild=False, verbose=False):
    configfn = os.path.splitext(cache_fn)[0] + "_cfg.pickle"
    if not rebuild and os.path.exists(configfn):
        with open(configfn, "rb") as fcfg:
            stored_cfg = pickle.load(fcfg)
            try:
                from numpy.testing import assert_equal
                assert_equal(cfg, stored_cfg)

                print(f"Using cached {cache_fn}")
                with open(cache_fn, "rb") as f:
                    return pickle.load(f)
            except:
                print(f"Found {configfn}, but cache needs rebuild.")
                if verbose:
                    print(f"Cached config: {stored_cfg}")
                    print(f"Current config: {cfg}")
    r = func()

    with open(cache_fn, "wb") as f:
        pickle.dump(r, f)
    with open(configfn, "wb") as f:
        pickle.dump(cfg, f, pickle.HIGHEST_PROTOCOL)

    return r


def get_file_len(f):
    pos=f.tell()
    eof_pos=f.seek(0,2)
    f.seek(pos)
    return eof_pos

def cached_iterator(path,cfg,iteratorfn,cache_tag='cache'):
    dir, file = os.path.split(path)
    file, ext = os.path.splitext(file)
    path_noext, _ = os.path.splitext(path)

    resultfn = path_noext + f".{cache_tag}.pickle"
    configfn = path_noext + f".{cache_tag}.cfg.pickle"
    
    print(f'looking for {configfn} with config: {cfg}')
    
    if os.path.exists(configfn):
        with open(configfn, "rb") as fcfg:
            stored_cfg, count = pickle.load(fcfg)
            if np.array_equal(stored_cfg, cfg):
                with open(resultfn, "rb") as f:
                    pbar = tqdm.trange(count)
                    for i in pbar:
                        yield pickle.load(f)
                return
            else:
                print(f'Rejecting cache data: Config does not match: f{stored_cfg}')
        os.remove(configfn) 
                
    with open(resultfn,"wb") as f:
        for i,data in enumerate(iteratorfn):
            pickle.dump(data, f)
            yield data

    count = i+1
    with open(configfn, "wb") as fcfg:
        pickle.dump((cfg,count),fcfg)
    

# Using numpy save/load. Pickles above 4GB fail, so this is a way around. func() is supposed to return a tuple of numpy arrays
def read_npy(path, cache_tag, cfg, func, rebuild=False, report=True):
    _, file = os.path.split(path)
    file, ext = os.path.splitext(file)

    path_noext, _ = os.path.splitext(path)

    resultfn = path_noext + f".{cache_tag}.npy"
    configfn = path_noext + f".{cache_tag}.cfg.pickle"
    if not rebuild and os.path.exists(configfn):
        with open(configfn, "rb") as fcfg:
            stored_cfg, numarrays = pickle.load(fcfg)
            if stored_cfg == cfg:
                # print(f'Using cached {resultsfn}')

                r = [0] * numarrays
                with open(resultfn, "rb") as f:
                    for k in range(numarrays):
                        r[k] = np.load(f)
                return tuple(r)

    r = func(path, cfg)

    with open(resultfn, "wb") as f:
        for k in range(len(r)):
            if report:
                print(f"{resultfn} Saving array {k} ({r[k].shape}, {r[k].dtype})")
            np.save(f, r[k])

    with open(configfn, "wb") as f:
        pickle.dump((cfg, len(r)), f)

    return r
