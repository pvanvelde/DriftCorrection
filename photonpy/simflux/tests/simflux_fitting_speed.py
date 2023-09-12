
import sys

import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.ticker import MaxNLocator


import matplotlib as mpl
#mpl.use('svg')
new_rc_params = {
#    "font.family": 'Times',
    "font.size": 15,
#    "font.serif": [],
    "svg.fonttype": 'none'} #to store text as text, not as path
mpl.rcParams.update(new_rc_params)


import photonpy.smlm.util as su
from photonpy.cpp.context import Context

import photonpy.cpp.gaussian as gaussian
from photonpy.cpp.estim_queue import EstimQueue
from photonpy.cpp.simflux import SIMFLUX
from photonpy.cpp.estimator import Estimator
import tqdm

def estimate_speed(thetas, repeats, psf:Estimator, ctx):
    nspots = repeats*len(thetas)
    sys.stdout.write(f"Measuring time it takes to fit {nspots} spots. Roisize: {psf.sampleshape[-1]}")

    q = EstimQueue(psf, batchSize=4*1024)
    smp = psf.GenerateSample(thetas)
    
    for i in range(repeats//5):
        q.Schedule(smp)
    q.Flush()
    q.WaitUntilDone()

    t0 = time.time()
    for i in range(repeats):
        q.Schedule(smp)
    q.Flush()
    q.WaitUntilDone()
    
    t1 = time.time()
    dt = t1-t0
    
    print(f": {nspots//dt} fits/s")
    
    q.Destroy()
    return nspots/dt

pitch = 221/65
k = 2*np.pi/pitch

mod = np.array([
           [0, k,   0.95, 0, 1/6],
           [k, 0, 0.95, 0, 1/6],
           [0, k,   0.95, 2*np.pi/3, 1/6],
           [k, 0, 0.95, 2*np.pi/3, 1/6],
           [0, k,   0.95, 4*np.pi/3, 1/6],
           [k, 0, 0.95, 4*np.pi/3, 1/6]
          ])

def drawfig(roisizes,sf_speed,g2d_speed):
    plt.figure()
    plt.plot(roisizes, g2d_speed, 'o-', label="2D Gaussian fit",linewidth=3,markersize=10)
    plt.plot(roisizes, sf_speed, 'o-', label="6-frame SIMFLUX fit", linewidth=3,markersize=10)
    plt.yscale('log')
    plt.xlabel('Region of interest size [pixels]')
    plt.ylabel('Fitting speed [fits/s]')
    plt.legend()
    plt.grid(True)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))    

with Context() as ctx:
    g = gaussian.Gaussian(ctx)

    sigma=2
    pixelsize=65
    numspots = 1024*8
    
    useCuda=True

    roisizes = np.arange(6,17,1)

    if True:    
        sf_speed = np.zeros(len(roisizes))
        g2d_speed = np.zeros(len(roisizes))
    
        for i,roisize in enumerate(roisizes):
            theta=[[roisize/2, roisize/2, 1000, 5]]
            theta=np.repeat(theta,numspots,0)
            theta[:,[0,1]] += np.random.uniform(-pitch/2,pitch/2,size=(numspots,2))
            
            #    with g.CreatePSF_XYZIBg(roisize, calib, True) as psf:
            g_psf = g.CreatePSF_XYIBg(roisize, sigma, useCuda)
                            
            s = SIMFLUX(ctx)
            sf_psf = s.CreateSIMFLUX2D_Gauss2D_PSF(sigma, mod, roisize, len(mod), True)
    
            g2d_speed[i] = estimate_speed(theta, 100, g_psf, ctx)
            sf_speed[i] = estimate_speed(theta, 10, sf_psf, ctx)


    drawfig(roisizes,sf_speed,g2d_speed)
