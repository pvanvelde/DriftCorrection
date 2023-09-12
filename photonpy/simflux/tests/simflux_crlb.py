import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time

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

from photonpy.cpp.simflux import SIMFLUX
from photonpy.cpp.estimator import Estimator
import tqdm

def estimate_precision(psf:Estimator, thetas, photons, smpfn=None):
    prec = np.zeros((len(photons),thetas.shape[1]))
    bias = np.zeros((len(photons),thetas.shape[1]))
    crlb = np.zeros((len(photons),thetas.shape[1]))
    Iidx = psf.ParamIndex('I')
    
    for i in tqdm.trange(len(photons)):
        thetas_ = thetas*1
        thetas_[:, Iidx] = photons[i]
        roipos = np.random.randint(0,20,size=(len(thetas_), psf.indexdims))
        roipos[:,0] = 0
        #roipos[:,0] %= 6
        if smpfn is not None:
            smp = smpfn(thetas_,roipos)
            #su.imshow_hstack(ev2[0], title='correct exp val' )
        else:
            smp = psf.GenerateSample(thetas_,roipos=roipos)
        estim,diag,traces = psf.Estimate(smp,roipos=roipos)
            
        crlb_ = psf.CRLB(thetas_,roipos=roipos)
        err = estim-thetas_
        prec[i] = np.std(err,0)
        bias[i] = np.mean(err,0)
        crlb[i] = np.mean(crlb_,0)

#    print(f'sigma bias: {bias[:,4]}')        
    return prec,bias,crlb

pitch = 221/65
k = 2*np.pi/pitch

mod = np.array([
           [0, k, 0,  0.95, 0, 1/6],
           [k, 0, 0, 0.95, 0, 1/6],
           [0, k, 0,  0.95, 2*np.pi/3, 1/6],
           [k, 0, 0, 0.95, 2*np.pi/3, 1/6],
           [0, k, 0,  0.95, 4*np.pi/3, 1/6],
           [k, 0, 0, 0.95, 4*np.pi/3, 1/6]
          ])


with Context() as ctx:
    g = gaussian.Gaussian(ctx)

    sigma=1.8
    roisize=10
    pixelsize=65
    numspots = 10000
    theta=[[roisize/2, roisize/2, 1000,8]]
    theta=np.repeat(theta,numspots,0)
    theta[:,[0,1]] += np.random.uniform(-pitch/2,pitch/2,size=(numspots,2))
    
    useCuda=True
    
#    with g.CreatePSF_XYZIBg(roisize, calib, True) as psf:
    g_psf = g.CreatePSF_XYIBg(roisize, sigma, useCuda)
                    
    s = SIMFLUX(ctx)
    sf_psf = s.CreateEstimator_Gauss2D(sigma, mod, roisize, len(mod), True)
    
    photons = np.logspace(2, np.log(5000)/np.log(10), 10)
    
    def noisyIntensitySample(noisefactor=1):
        def _noisyIntensitySample(thetas,roipos):
            x = thetas[:,0] + roipos[:,2]
            y = thetas[:,1] + roipos[:,1]
            exc = mod[:,4]*(1+mod[:,2]*np.sin(mod[:,0]*x[:,np.newaxis]+mod[:,1]*y[:,np.newaxis]-mod[:,3]))
            
            g_theta = np.repeat(thetas, len(mod), axis=0)
            g_theta[:,2] += np.random.normal(0,noisefactor*np.sqrt(g_theta[:,2]),size=len(g_theta))
            g_theta[:,2] *= exc.flatten()
            g_theta[:,3] /= len(mod)
            
            smp = g_psf.GenerateSample(g_theta,roipos=roipos[:,1:].repeat(len(mod),0))
            smp = np.reshape(smp,(len(thetas),len(mod),roisize,roisize))
            
            return smp
        return _noisyIntensitySample

    if False:        
        th=np.array([[5,5,500,1.0]])
        ev=sf_psf.ExpectedValue(th)
        su.imshow_hstack(ev[0], 'sf')
        
        ev2=noisyIntensitySample(th,np.zeros((1,3)))
        su.imshow_hstack(ev2[0],'nev')


    axes=['x','I']
    unit_scale=[pixelsize,1]
    yticks=[
            [1,2,5,10,20,50],
            [20,30,50,70,100]
            ]
    xticks=[100,200,500,1000,2000,5000]
    unit=['nm','photons'] 
    if True:
        data = [
            #(sf_psf, False, "SIMFLUX + Intensity Noise", estimate_precision(sf_psf, theta, photons, smpfn=noisyIntensitySample(1))),
            (sf_psf, True, "SIMFLUX", estimate_precision(sf_psf, theta, photons)),
            (g_psf, True, "", estimate_precision(g_psf, theta, photons)),
            #(g_z_psf, 'Z-Fitted astig. 2D Gauss',estimate_precision(g_z_psf, theta_z, photons)),
            #(g_s_psf, '2D Gauss Sigma', estimate_precision(g_s_psf, theta_sig[:,:5], photons)),
            #(g_sxy_psf, '2D Gauss Sigma XY', estimate_precision(g_sxy_psf, theta_sig, photons)),
            ]

        for i,ax in enumerate(axes):
            fig=plt.figure()
            for psf,plot_crlb,name,(prec,bias,crlb) in data:
                ai = psf.ParamIndex(ax)
                plt.plot(photons,unit_scale[i]*prec[:,ai],'o',label=f'MLE {name}', linewidth=3)
                if plot_crlb: 
                    plt.plot(photons,unit_scale[i]*crlb[:,ai],'--', label=f'CRLB {name}', linewidth=3)

            if i==1:
                plt.plot(photons,np.sqrt(photons),'--', label=r'$\sqrt{N}$')

            plt.ylabel(f'Precision [{unit[i]}]' )
            plt.xlabel(f'Emitter intensity $N$ [photons]' )
            plt.title(f'{ax} axis')
            gca=plt.gca()
            plt.xscale("log")
            plt.yscale("log")
            
            gca.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            gca.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            gca.set_yticks(yticks[i])
            gca.set_yticks([],minor=True)
            gca.set_xticks(xticks)
            plt.grid(True)
            plt.legend()
            plt.savefig(f'plots/{ax}_crlb.svg')

            plt.figure()
            for psf,plot_crlb,name,(prec,bias,crlb) in data:
                ai = psf.FindThetaIndex(ax)
                plt.plot(photons,bias[:,ai],label=f'Bias {name}')

            plt.title(f'{ax} axis')
            plt.grid(True)
            plt.xscale("log")
            
            gca=plt.gca()
            gca.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            gca.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            gca.set_xticks([100,300,1000,3000,10000])
            plt.legend()
            plt.show()

    if False:
        for ax in [0,2]:
            plt.figure()
            theta_z = np.zeros((numspots,5))
            theta_z[:,0:4] = theta
            addplot(ax,g_z_psf,theta_z,photons, '2D Gaussian (xyz) MLE')

            axname = g_z_psf.param_names[ax]
            plt.title(f'Axis {axname}')
            plt.xscale("log")
            plt.yscale("log")
            plt.grid(True)
            plt.legend()
            plt.show()
    
    if False:
        sigphotons = 1000
        noiselist = np.linspace(0,4,20)
        prec = np.zeros((len(noiselist),4))
        for i,noiselev in enumerate(noiselist):
            prec[i],_,_=estimate_precision(sf_psf, theta, [sigphotons], smpfn=noisyIntensitySample(noiselev))

        theta[:,2] = sigphotons
        g_crlb = np.mean( g_psf.CRLB(theta), 0 )
        plt.figure()
        plt.plot(noiselist*np.sqrt(sigphotons), g_crlb[0]/prec[:,0])
        plt.xlabel('Standard deviation of added emitter intensity noise [photons]')
        plt.ylabel('Improvement factor in X')            
        
        plt.figure()
        plt.plot(noiselist*np.sqrt(sigphotons), prec[:,0] * unit_scale[0], label='MLE SIMFLUX with added intensity noise')
        plt.plot(noiselist*np.sqrt(sigphotons), np.ones(len(noiselist))*g_crlb[0] * unit_scale[0], label='CRLB SMLM')
        plt.xlabel('Standard deviation of added emitter intensity noise [photons]')
        plt.ylabel('Precision in X [nm]')
        plt.legend()