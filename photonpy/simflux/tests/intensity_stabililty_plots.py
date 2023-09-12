import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.signal


debugMode = False

sys.path.append("../..")

import tifffile
from smlmlib.base import SMLM
from smlmlib.context import Context
from smlmlib.gaussian import Gaussian
from smlmlib.psf_queue import PSF_Queue, PSF_Queue_Results
from smlmlib.simflux import SIMFLUX
import smlmlib.spotdetect as spotdetect
from smlmlib.calib import GainOffset_Calib,GainOffsetImage_Calib
import smlmlib.util as su
import utils.localizations as loc
import tqdm
import time
import threading
from utils.caching import cached_iterator
from smlmlib.image_filter_queue import BlinkFilterQueue, BlinkDetectorQueueConfig
import pickle

from simflux.read_tiff import tiff_get_image_size, tiff_read_file, tiff_get_movie_size, get_tiff_mean

import matplotlib as mpl
#mpl.use('svg')
new_rc_params = {
#    "font.family": 'Times',
    "font.size": 12,
#    "font.serif": [],
    "svg.fonttype": 'none'} #to store text as text, not as path
mpl.rcParams.update(new_rc_params)


import scipy.io

def load_mod(tiffpath):
    with open(os.path.splitext(tiffpath)[0]+"_mod.pickle", "rb") as pf:
        mod = pickle.load(pf)['mod']
        return mod
    
    
def gauss_filter_kernel(sigma, window):
    x = np.arange(-window//2,window//2)
    return x*np.exp(-(x**2)/(2*sigma**2))

def filter_results(r,roisize):
    minX = 2.1
    minY = 2.1
    r.FilterXY(minX,minY,roisize-minX-1, roisize-minY-1)
    r.Filter(np.where(r.iterations<90)[0])
    

def localize_g2d(data,spots,sigma,ctx):
    roisize=data.shape[-1]
    roi = np.zeros((len(data),roisize,roisize))
    numframes = spots['numframes']
    print(f"Summing {len(data)} spots" )
    for i in range(len(data)):
        roi[i] = np.sum(data[i,:numframes[i]],0)

    print(f"Running 2D Gaussian fits")        
    cx,cy = spots['cx'],spots['cy']
    roipos=np.zeros((len(data),2),dtype=np.int32)
    roipos[:,0]=cy
    roipos[:,1]=cx
    g=Gaussian(ctx)
    psf=g.CreatePSF_XYIBg(roi.shape[2],sigma,True)
    q = PSF_Queue(psf, batchSize=1024)
    q.Schedule(roi,roipos,np.arange(len(data)))
    q.WaitUntilDone()
    r = q.GetResults()
    r.SortByID(isUnique=True)
    filter_results(r,roisize)
    return r


def localize_simflux(roi,spots,sigma,mod,ctx):
    print(f"Running SIMFLUX fits")
    sf=SIMFLUX(ctx)
    psf=sf.CreateSIMFLUX2D_Gauss2D_PSF(sigma,mod,roi.shape[3],roi.shape[1],simfluxEstim=True,defineStartEnd=True)
    const = np.zeros((len(roi),2),dtype=np.float32)
    const[:,0] = 0
    const[:,1] = spots['numframes']
    
    roipos=np.zeros((len(roi),3),dtype=np.int32)
    roipos[:,0] = spots['startframe'] % len(mod)
    roipos[:,1] = spots['cy']
    roipos[:,2] = spots['cx']
    q = PSF_Queue(psf, batchSize=1024)
    q.Schedule(roi, roipos, np.arange(len(roi)), const)
    q.WaitUntilDone()
    r = q.GetResults()
    r.SortByID(isUnique=True)
    filter_results(r,roi.shape[-1])
    return r


class TiffReaderThread(threading.Thread):
    def __init__(self,*args):
        super(TiffReaderThread, self).__init__()
        self.args=args

    def run(self):
        fn,q,maxframes = self.args

        if maxframes is None:
            maxframes=-1

        for idx,f in tiff_read_file(fn, 0, maxframes):
            q.PushFrame(f)



def detect_spots(ctx, fn, offset, roisize, roilen, maxframes=None):
    imgshape, nframes = tiff_get_movie_size(fn)
    print(f"TIFF contains {nframes} frames")

    if type(offset) == np.ndarray:
        gain = np.ones(imgshape)*2.2
        calib = GainOffsetImage_Calib(gain, offset, ctx)
    else:
        calib = None
        
    maxFilterT = 10
    cfg = BlinkDetectorQueueConfig.make(imgshape=imgshape, uniformFilter1=5,uniformFilter2=15, 
                roisize=roisize, maxFilterT=maxFilterT,maxFilterXY=10, 
                startThreshold=startThreshold, endThreshold=startThreshold*.8, maxROIframes=roilen, maxPeakDist=5)

    q = BlinkFilterQueue(cfg, tfilter, calib, ctx)

    if maxframes is not None:
        nframes = maxframes
        
    tiffReaderThread = TiffReaderThread(fn,q,maxframes)
    tiffReaderThread.start()
    
    t0 = time.time()
    while True:
        if q.NumFinishedFrames()>=nframes-len(tfilter)-maxFilterT-3:
            break
        
        if q.GetSpotCount()>100:
            yield q.GetSpots()
            
        time.sleep(0.01)
        
    framenum = q.NumFinishedFrames()
    t1 = time.time()
    tiffReaderThread.join()
    dt = t1-t0
    print(f"fps: { framenum//dt}" )

def plot_moving_windows(estim,ibg,rois,numframes):
    mw_results=[]

    mwindowsizes = np.arange(2,15)
    for mwindow in mwindowsizes:
        mw_var = []
        mw_mean = []

        selected = []
        intensity_var = []
        intensity_mean = []
    
        photons_var = []
        photons_mean = []
    
        hitborder = (estim[:,1]<3) | (estim[:,0]<3) | (estim[:,1]>roisize-3) | (estim[:,0]>roisize-3)
    
        plot_traces = False
        
        if plot_traces:    
            plt.figure()
            plt.ylabel('Emitter intensity [photons]')
            plt.xlabel('Time [frames]')
            plt.title('Intensity traces of selected spots')
        for s in range(len(ibg)):
            if numframes[s]<18 or estim[s,4]<1.7 or estim[s,4]>2:
                continue
            
            # roi region
            if hitborder[s]:
                continue
            
            spot_i = ibg[s,2:numframes[s]-1]
            spot_photons = np.sum(rois[s,2:numframes[s]-1],(-1,-2))
            if np.min(spot_i[:,0])>200 and np.mean(spot_i[:,0])<1500:
                selected.append(s)
                intensity_var.append( np.var(spot_i,0) )
                intensity_mean.append( np.mean(spot_i,0) )
                
                X,Y=np.meshgrid(np.arange(mwindow),np.arange(len(spot_i)-mwindow))
                mwindices = X+Y
                mw_spot_i = spot_i[mwindices,0]
                
                # unbiased estimate over the moving window of 6 frames
                mw_var.append(np.var(mw_spot_i,axis=1,ddof=1))
                mw_mean.append(np.mean(mw_spot_i,axis=1))
    
                photons_var.append(np.var(spot_photons))
                photons_mean.append(np.mean(spot_photons))
                
                if plot_traces and len(selected)<40:
    #               su.imshow_hstack(rois[s,0:spotnumframes[s]])
                    #print(f"{s}: mean{np.mean(spot_i,0)} var:{np.var(spot_i,0)} sigma={estim[s,4]}")
    #                plt.plot(np.arange(len(spot_i)), spot_i)
                    plt.plot(np.arange(len(spot_photons)), spot_i[:,0])
        
        print(f"Selected {len(selected)} spots")
            
        intensity_mean = np.array(intensity_mean)
        intensity_var = np.array(intensity_var)
        
        photons_var = np.array(photons_var)
        photons_mean = np.array(photons_mean)
    
        if mwindow==2:
            plt.figure()
            np.random.shuffle(selected)
            for i in range(np.minimum(len(selected),10)):
        #        su.imshow_hstack(rois[selected[i],2:numframes[selected[i]]-1])
                su.imshow_rois(rois[selected[i],2:numframes[selected[i]]-1])
    
        mw_var = np.concatenate(mw_var)
        mw_mean = np.concatenate(mw_mean)
    
        print(f'Variance moving window ({mwindow}-frame window): median={np.median(mw_var)} mean={np.mean(mw_var)}')
        
        mw_results.append([np.median(mw_var),np.mean(mw_mean)])
    
        plt.figure()
        plt.subplot(211)
        plt.title(f'{mwindow}-frame sets')
        bins=plt.hist(mw_mean, bins=100, range=(0,8000))[1]
        plt.xlabel(f'Spot intensity means of {mwindow}-frame sets')
        plt.subplot(212)
        plt.hist(mw_var,bins=bins)
        plt.xlabel(f'Spot intensity variance of {mwindow}-frame sets')


        
    plt.figure()
    plt.plot(mwindowsizes,np.array(mw_results)[:,0],label='Median variance within window')
    plt.plot(mwindowsizes,np.array(mw_results)[:,1],label='Mean spot intensity')
    plt.title('Median intensity variance in moving windows')
    plt.xlabel('Window size [frames]')
    plt.ylabel('Variance [photons]')
    plt.legend()
    
    return mw_results


def analyse_traces(roisize,sigma,spot_source,ctx):
    g = Gaussian(ctx)
    psf=g.CreatePSF_XYIBgSigma(roisize,sigma,True)

    spot_list=[]
    roi_list=[]
    for spots,rois in spot_source:
        spot_list.append(spots)
        roi_list.append(rois)
        
    spots = np.concatenate(spot_list)
    rois = np.concatenate(roi_list)

    startframe=spots['startframe']
    numframes=spots['numframes']
    cx,cy = spots['cx'],spots['cy']
    roipos=np.zeros((len(rois),2),dtype=np.int32)
    roipos[:,0]=cy
    roipos[:,1]=cx

    numspots=len(numframes)
    summed = np.zeros((numspots,roisize,roisize))
    for i in range(numspots):
        summed[i] = np.sum(rois[i,:numframes[i]],0)
        
    estim = psf.ComputeMLE(summed)[0]
    crlb_sum = psf.CRLB(estim)
    
    print(estim[:,4])

    estim_rep = np.repeat(estim, rois.shape[1], 0)
    imgs = rois.reshape((rois.shape[0]*rois.shape[1],roisize,roisize))
    ibg,ibg_crlb = g.EstimateIBg(imgs, estim_rep[:,4], estim_rep[:,[0,1]],useCuda=True)
    ibg = ibg.reshape((rois.shape[0],rois.shape[1],2))
    ibg_crlb = ibg_crlb.reshape((rois.shape[0],rois.shape[1],2))
    
    #mw_results=plot_moving_windows(estim,ibg,rois,numframes)
    
    os.makedirs('plots',exist_ok=True)
    
    selected = []
    intensity_var = []
    intensity_mean = []
    crlb_mean = []

    photons_var = []
    photons_mean = []

    f2f = []
    f2f_mean = []
    f2f_var = []
    f2f_var_corrected = []
    for k in range(10):
        f2f_mean.append([])
        f2f_var.append([])
        f2f.append([])
        f2f_var_corrected.append([])
    
    hitborder = (estim[:,1]<3) | (estim[:,0]<3) | (estim[:,1]>roisize-3) | (estim[:,0]>roisize-3)

    plot_traces = False
    
    if plot_traces:    
        plt.figure()
        plt.ylabel('Emitter intensity [photons]')
        plt.xlabel('Time [frames]')
        plt.title('Intensity traces of selected spots')
    for s in range(len(ibg)):
        if numframes[s]<18 or estim[s,4]<1.7 or estim[s,4]>2:
            continue
        
        # roi region
        if hitborder[s]:
            continue
        
        spot_ibg = ibg[s,2:numframes[s]-1]
        spot_ibg_crlb = ibg_crlb[s,2:numframes[s]-1]
        estimation_variance = spot_ibg_crlb**2 - spot_ibg
        spot_photons = np.sum(rois[s,2:numframes[s]-1],(-1,-2))
        if np.min(spot_ibg[:,0])>200 and np.mean(spot_ibg[:,0])<1500:
            selected.append(s)
            intensity_var.append( np.var(spot_ibg,0) )
            intensity_mean.append( np.mean(spot_ibg,0) )
           
            crlb_mean.append(np.mean(spot_ibg_crlb,0))
            
            for k in range(len(f2f_mean)):

                d = spot_ibg[1+k:]-spot_ibg[0:-1-k] # 2x the variance
                var_wnd = np.var(spot_ibg[:,0],ddof=1)
                estvar = estimation_variance[1+k:]+estimation_variance[0:-1-k]
                f2f[k].append(d)
                f2f_mean[k].append(np.mean(d,0)[0])
#                var = np.var(d[:,0],ddof=0)
                var = np.mean((d**2)/2,0)[0]
                f2f_var[k].append(var)
                f2f_var_corrected[k].append(var-np.mean(estimation_variance,0)[0])

            photons_var.append(np.var(spot_photons))
            photons_mean.append(np.mean(spot_photons))
            
            if plot_traces and len(selected)<40:
                plt.plot(np.arange(len(spot_photons)), spot_i[:,0])
    
    print(f"Selected {len(selected)} spots")

    if False:
        np.random.shuffle(selected)
        for i in range(np.minimum(len(selected),10)):
    #        su.imshow_hstack(rois[selected[i],2:numframes[selected[i]]-1])
            su.imshow_hstack(rois[selected[i],2:numframes[selected[i]]-1])
            plt.savefig(f'plots/roi{i}.svg')
        
    intensity_mean = np.array(intensity_mean)
    intensity_var = np.array(intensity_var)
    
    photons_var = np.array(photons_var)
    photons_mean = np.array(photons_mean)
    
    crlb_mean = np.array(crlb_mean)

    for k in range(len(f2f)):
        f2f[k]=np.concatenate(f2f[k])

  #  for k in range(len(f2f_var)):
#        f2f_var[k]=np.concatenate(f2f_var[k])
 #       print(len(f2f_var[k]))

    plt.figure()
    plt.subplot(211)
    bins=plt.hist(intensity_mean[:,0], bins=50, range=(0,8000))[1]
    plt.xlabel('Spot intensity mean')
    plt.subplot(212)
    plt.hist(intensity_var[:,0],bins=bins)
    plt.xlabel('Spot intensity variance')
    
    print(f'Spot mean I and background: {np.mean(intensity_mean,0)}' )
    
    plt.figure()
    plt.scatter(np.sqrt(intensity_mean[:,0]),crlb_mean[:,0],label= 'Mean per-spot CRLB')
    x = np.linspace(0,40)
    plt.plot(x,x,'--',label='sqrt(Nest)')
    plt.xlabel('sqrt(Nest)')
    plt.ylabel('crlb')
    plt.savefig('plots/crlb_vs_nphotons.svg')

    if False:
        for k in range(len(f2f_var)):
            var = np.array(f2f_var[k])
            means = np.array(f2f_mean[k])
            plt.figure(figsize=(6,8))
            plt.subplot(211)
            plt.hist(var, bins=50, range=(0,8000))
            plt.xlabel(f'Frame to frame spot intensity variance [step {k+1} difference]')
    
    #        plt.subplot(312)
     #       plt.hist(means, bins=50, range=(-10,10))
      #      plt.xlabel('Frame to frame spot intensity mean')
    
            plt.subplot(212)
            plt.hist(intensity_mean[:,0], bins=50, range=(0,8000))
            plt.xlabel('Spot intensity mean')
            plt.tight_layout()
            
            plt.savefig(f'plots/f2f-{k}.svg')
            plt.savefig(f'plots/f2f-{k}.png')

    def corrected_spot_variance():            
        plt.figure(figsize=(6,5))
        plt.subplot(311)
        plt.hist([ f2f_var[0], f2f_var[5]], bins=30, range=(0,8000))
        plt.legend(labels=['T=1', 'T=6'])
        plt.xlabel(f'Spot intensity variance')# as function of distance in time')
    
        plt.subplot(312)
        plt.hist([ f2f_var_corrected[0], f2f_var_corrected[5]], bins=30, range=(0,8000))
        plt.legend(labels=['T=1', 'T=6'])
        plt.xlabel(f'Corrected spot intensity variance (var - crlb^2)')# as function of distance in time')
    
        plt.subplot(313)
        plt.hist(intensity_mean[:,0], bins=50, range=(0,8000))
        plt.xlabel('Spot intensity mean')
        plt.tight_layout()
        
        plt.savefig(f'plots/f2f-1and6-corrected.svg')
        plt.savefig(f'plots/f2f-1and6-corrected.png')

    def spot_crlb_hist(T):
        plt.figure(figsize=(6,5))
        plt.subplot(311)
        plt.hist([ f2f_var[0], f2f_var[T-1]], bins=30, range=(0,8000))
        plt.legend(labels=['T=1', f'T={T}'])
        plt.xlabel(f'Spot intensity variance [photons^2]')# as function of distance in time')
    
        plt.subplot(312)
        plt.hist(crlb_mean[:,0]**2, bins=50, range=(0,8000))
        plt.xlabel('Spot mean CRLB^2 [photons^2]')
        plt.tight_layout()

        plt.subplot(313)
        plt.hist(intensity_mean[:,0], bins=50, range=(0,8000))
        plt.xlabel('Spot mean intensity [photons]')
        plt.tight_layout()
        
        plt.savefig(f'plots/f2f-1and{T}.svg')
        plt.savefig(f'plots/f2f-1and{T}.png')

    corrected_spot_variance()
    corrected_spot_variance()
    spot_crlb_hist(6)
    spot_crlb_hist(10)

    plt.figure()
    plt.subplot(211)
    bins=plt.hist(intensity_mean[:,1], bins=50)[1]
    plt.xlabel('Spot background mean')
    plt.subplot(212)
    plt.hist(intensity_var[:,1],bins=bins)
    print(intensity_var[:,1])
    plt.xlabel('Spot background variance')

    if False:
        plt.figure()
        plt.subplot(211)
        bins=plt.hist(photons_mean, bins=50, range=(0,8000))[1]
        plt.xlabel('ROI photon counts mean')
        plt.subplot(212)
        plt.hist(photons_var,bins=bins)
        plt.xlabel('ROI photon counts variance')
    
    if True:
        f2f_var = np.array(f2f_var)
        f2f_mean = np.array(f2f_mean)
        f2f_var_corrected =  np.array(f2f_var_corrected)
        plt.figure(figsize=(5,5))
        spot_intensity_mean = np.median(intensity_mean[:,0],0)
        print(spot_intensity_mean )
        plt.plot(np.arange(10)+1,np.median(f2f_var,1),'-o',
                 label='Median of two-frame spot intensity variance')
        plt.plot(np.arange(10)+1,spot_intensity_mean*np.ones(10),'-o', 
                 label='Median of spot intensity means')
        plt.title('Frame to frame intensity variance vs separation in time')
        plt.ylabel(r'Variance [$photons^2$]')
        plt.xlabel('Time separation [$frames$]')
        plt.legend()
        plt.savefig(f"plots/f2f_k_range.svg")

    if False:        
        plt.figure(figsize=(8,5))
        ax=plt.subplots(1,1)[1]
        
        xticks = [r'$CRLB^2$']
        ax.violinplot(vert=True,positions=[0],dataset=crlb_mean[:,0]**2,showmedians=True,showextrema=False)
        for i,k in enumerate(np.arange(0,10,2)):
            xticks.append(f"T={k+1}")
            maxvar=5000
            ax.violinplot(vert=True,positions=[i+1],dataset=f2f_var[k][f2f_var[k]<maxvar],showmedians=True,showextrema=False,widths=0.7)
    
        plt.ylabel('Variance [photons^2]')
        plt.xticks(np.arange(len(xticks)), xticks)
    #    plt.plot(np.arange(10)+1,spot_intensity_mean*np.ones(10),'ok', 
     #            label='Median of spot intensity means')
        plt.savefig(f"plots/violinplot.svg")

    plt.figure(figsize=(5,6))
    ax=plt.subplots(1,1)[1]
    
    
    def set_violinplot_colors(p):
        rrred = '#ff2222'
        bluuu = '#2222ff'
        for vp in p['bodies']:
            vp.set_facecolor(bluuu)
            vp.set_edgecolor(rrred)
            vp.set_linewidth(1)
            vp.set_alpha(0.5)

    vpx = []
    vpy = []
    
    xticks = [r'$CRLB$']

    vpx.append(0)
    vpy.append(crlb_mean[:,0])
    
    for i,k in enumerate(np.arange(0,10)):
        xticks.append(f"T={k+1}")
        maxstd=100
        data = np.sqrt(f2f_var[k])
                
        vpx.append(i+1)
        vpy.append(data[data<maxstd])
        
    for i,d in enumerate(vpy):
        print(f"{xticks[i]}: median = {np.median(d):.3f}; mean={np.mean(d):.3f}")
        
    vp=ax.violinplot(vert=True,positions=vpx,dataset=vpy,showextrema=False,widths=0.7)
    set_violinplot_colors(vp)
    ax.plot(vpx,[np.median(d) for d in vpy],'o',color='black')

    plt.ylabel('Standard deviation [photons]')
    plt.xticks(np.arange(len(xticks)), xticks,rotation=45)
#    plt.plot(np.arange(10)+1,spot_intensity_mean*np.ones(10),'ok', 
 #            label='Median of spot intensity means')
    plt.savefig(f"plots/violinplot-std.svg")
    
    scipy.io.savemat('intensity_mean_and_var.mat', 
                     {'means':intensity_mean,'var':intensity_var})

    print(f"selected: {len(selected)}")            
    return ibg, numframes, rois, spots, ibg_crlb, crlb_sum,estim

fn = 'C:/data/storm-wf/60 mw Pow COS7_aTub_A647_sulfite_10mM_MEA_50Glyc/1.tif'
#fn = '../../../../SMLM/data/hd/sim1_200mw_9_MMStack_Pos0.ome.tif'; startThreshold=10
#    fn = '../../../../SMLM/data/hd/sim1_200mw_7_MMStack_Pos0_merge.ome.tif'; startThreshold=10
#fn = '../../../../SMLM/data/sim-silm/object_wlc_15.tif'; startThreshold = 12;
#fn = '../../../../SMLM/data/sim4_1/sim4_1.tif'; startThreshold=30
#fn = '../../../../SMLM/data/noshift/1_1.tif'; startThreshold=30
#fn = 'C:/dev/simflux/data/7-17-2019 STORM COS/Pos1_WF_merge.tif';

#fn = '/data/noshift/2_1.tif'; startThreshold=30
#    fn = '../../../../SMLM/data/hd/1_3_MMStack_Pos0_1.ome.tif'; startThreshold=10
tsigma=3
tfilter=gauss_filter_kernel(tsigma,16)
offset = get_tiff_mean('/data/GainImages/1bg_1_MMStack_Pos0.ome.tif')
maxframes = None
roisize = 11
roilen = 80

with SMLM(debugMode=False) as smlm,Context(smlm) as ctx:
    spot_source = cached_iterator(fn, (maxframes,roisize,roilen), 
        detect_spots(ctx, fn, offset, roisize, roilen, maxframes), cache_tag='spots' )

    sigma = 1.83
    ibg, spotnumframes,rois,spots,ibg_crlb,sum_crlb,sum_estim = analyse_traces(roisize,sigma,spot_source,ctx)

