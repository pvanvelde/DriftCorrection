# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from photonpy.cpp.image_proc import RollingMedianImageFilter
from photonpy import Context

import tqdm

import time

def view_movie(mov):
    import napari    
    
    with napari.gui_qt():
        napari.view_image(mov)



imgshape = (300,300)

with Context(debugMode=False) as ctx:
    mf = RollingMedianImageFilter(imgshape, 100, ctx)

    n = 5000
    
    data = np.random.poisson(50, size=(21, *imgshape))
    result = []
    
    with tqdm.tqdm(total=n) as pb:
        
        for i in range(n):
            while mf.GetQueueLength() > 50:
                time.sleep(0.02)
            
            mf.PushFrameF32(data[i % len(data)])
            
            if mf.NumFinishedFrames()>0:
                result.append(mf.ReadResultFrame()[1])
                pb.update(1)
            
        while len(result) != n:
            
            if mf.NumFinishedFrames()>0:
                result.append(mf.ReadResultFrame()[1])
                pb.update(1)

    view_movie(np.array(result))
