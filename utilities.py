import pickle
from filelock import FileLock
import os
import time

import numpy as np
import scipy.special

class timing:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, *args):
        print(self.name, "took", time.time() - self.start, "seconds")

def logsumexp(x, y=None, axis=None):
    if y is None:
        return scipy.special.logsumexp(x, axis=axis)
    else:
        return scipy.special.logsumexp(np.array([x,y]), axis=axis)

# possibly stale versions of cashed dictionaries
possibly_stale = {}
def stale_cached_dictionary_lookup(fn):
    global possibly_stale
    
    if fn in possibly_stale:
        return possibly_stale[fn]
    else:
        return {}

updates_until_next_flush = {}
def buffered_extend_cached_dictionary(fn, k, v):
    global possibly_stale, updates_until_next_flush

    if fn not in updates_until_next_flush:
        updates_until_next_flush[fn] = 50
    
    flush = updates_until_next_flush[fn] <= 0

    possibly_stale[fn][k] = possibly_stale[fn].get(k, [])
    possibly_stale[fn][k].extend(v)

    if not flush:
        updates_until_next_flush[fn] -= 1
        return possibly_stale[fn][k]
    
    with FileLock(f'{fn}.lck'):
        db = possibly_stale[fn]
            
        if k not in db:
            db[k] = []
        db[k].extend(v)

        temporary_file = f"/tmp/{time.time()}.pickle"
        with open(temporary_file, 'wb') as pfile:
            pickle.dump(db, pfile)
            
        os.system(f"mv {temporary_file} {fn}")
            
    return db[k]


def atomically_read_cached_dictionary(fn, k):
    global possibly_stale
    
    with FileLock(f'{fn}.lck'):
        if not os.path.exists(fn): return None
        
        with open(fn, 'rb') as pfile:
            db = pickle.load(pfile)
            possibly_stale[fn] = db
            
            if k not in db:
                return []
            return db[k]



def atomically_extend_cached_dictionary(fn, k, v):
    global possibly_stale
    
    with FileLock(f'{fn}.lck'):
        if not os.path.exists(fn):
            print("Creating", fn, "for the first time")
            with open(fn, "wb") as handle:
                pickle.dump({k:v}, handle)
            return v
        
        with open(fn, 'rb') as pfile:
            db = pickle.load(pfile)
            
        if k not in db:
            db[k] = []
        db[k].extend(v)

        temporary_file = f"/tmp/{time.time()}.pickle"
        with open(temporary_file, 'wb') as pfile:
            pickle.dump(db, pfile)
            
        os.system(f"mv {temporary_file} {fn}")

    possibly_stale[fn] = db
            
    return db[k]
