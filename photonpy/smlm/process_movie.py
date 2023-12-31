

import numpy as np
import time
import tqdm
import os

from photonpy import Context, GaussianPSFMethods, Estimator, Dataset
import photonpy.cpp.spotdetect as spotdetect
from photonpy.cpp.calib import GainOffset_Calib
from photonpy.cpp.calib import GainOffsetImage_Calib
from photonpy.cpp.estim_queue import EstimQueue
from photonpy.cpp.roi_queue import ROIQueue, ROIInfoDType
from photonpy.cpp.image_proc import ROIExtractor
import photonpy.utils.multipart_tiff as tiff
import tifffile
import matplotlib.pyplot as plt

from photonpy.utils.array import peek_first


from scipy.interpolate import InterpolatedUnivariateSpline

def end_of_file(f):
    curpos = f.tell()
    f.seek(0,2)
    file_size = f.tell()
    f.seek(curpos,0)
    return curpos == file_size 

def detect_spots_slow(sdcfg, calib, movie, sumframes, output_fn, batch_size, ctx:Context):
    sm = spotdetect.SpotDetectionMethods(ctx)

    with open(output_fn, "wb") as f:
        roishape = [sumframes, sdcfg.roisize,sdcfg.roisize]
        
        numframes = 0
        numrois = 0
        nsummed = 0
        
        batch_info = []
        batch_rois = []
        
        rois_in_batch = 0

        def save_rois():
            nonlocal numrois, rois_in_batch
            np.save(f, np.concatenate(batch_info), allow_pickle=False)
            np.save(f, np.concatenate(batch_rois), allow_pickle=False)
            numrois += len(rois_info)
            rois_in_batch = 0
            batch_info.clear()
            batch_rois.clear()
        
        framebuf = None        
        for i,img in enumerate(movie):
            if framebuf is None:
                framebuf = np.zeros((sumframes, img.shape[0], img.shape[1]), dtype=img.dtype)
            
            framebuf[nsummed] = img
            nsummed += 1

            
            if nsummed == sumframes:
                sum_rois, cornerYX, scores, spotz = sm.ProcessFrame(framebuf.sum(0), sdcfg, sdcfg.roisize, 200, calib)
                cornerPosZYX = np.zeros((len(sum_rois),3),dtype=np.int32)
                cornerPosZYX[:,1:] = cornerYX
                
                for j in range(len(framebuf)):
                    framebuf[j] = calib.process_frame(framebuf[j])
                    
                rois = sm.ExtractROIs(framebuf, roishape, cornerPosZYX)
                nsummed = 0
                
                rois_info = np.zeros((len(sum_rois)),dtype=ROIInfoDType)
                rois_info ['id'] = i//sumframes
                rois_info ['score'] = scores
                rois_info ['z'] = spotz
                rois_info ['y'] = cornerYX[:,0]
                rois_info ['x'] = cornerYX[:,1]
                
                batch_info.append(rois_info)
                batch_rois.append(rois)
                
                numrois += len(sum_rois)
                rois_in_batch += len(sum_rois)
                
                if rois_in_batch >= batch_size:
                    save_rois()
                           
            numframes += 1
            
        return numrois, numframes
    

def detect_spots(sdcfg, calib, movie, sumframes, output_fn, batch_size, ctx:Context, numThreads=3):
    sm = spotdetect.SpotDetectionMethods(ctx)

    with Context(ctx.smlm) as lq_ctx, open(output_fn, "wb") as f:
        roishape = [sdcfg.roisize,sdcfg.roisize]
        
        numframes = 0
        numrois = 0
        
        for i,img in enumerate(movie):
            if i==0:
                q,rq = sm.CreateQueue(img.shape, roishape, sdcfg, calib=calib,sumframes=sumframes, ctx=lq_ctx, numThreads=numThreads)
        
            def save_rois(rois_info, pixels):
                np.save(f, rois_info, allow_pickle=False)
                np.save(f, pixels, allow_pickle=False)
                nonlocal numrois
                numrois += len(rois_info)

            q.PushFrameU16(img)
            numframes += 1
            
            rl = rq.Length()
            if rl>batch_size:
                save_rois(*rq.Fetch())
                   
        while q.NumFinishedFrames() < numframes//sumframes:
            time.sleep(0.1)
        
        if rq.Length()>0:
            save_rois(*rq.Fetch())
            
        return numrois, numframes


def load_rois_iterator(rois_fn, maxrois=None):
    """
    Load rois sequentially so we can deal with very large datasets
    """
    with open(rois_fn, "rb") as f:
        total = 0
        while not end_of_file(f):
            rois_info = np.load(f,allow_pickle=True)
            pixels = np.load(f,allow_pickle=True)
            
            if maxrois is not None:
                if len(pixels) + total >= maxrois:
                    rem = maxrois - total
                    yield rois_info[:rem], pixels[:rem]
                    return

            total += len(pixels)
            yield rois_info, pixels


def load_rois(rois_fn, maxrois=None):
    rois_info = []
    pixels = []
    for ri,px in load_rois_iterator(rois_fn, maxrois):
        rois_info.append(ri)
        pixels.append(px)

    return np.concatenate(rois_info), np.concatenate(pixels)


def extract_rois_iterator(movie, roipos, roiframe, calib, roisize, 
                          ctx:Context, minBatchSize):
    """
    Extract predefined ROIs from a TIFF or other image iterator. 
    Calib: camera calibration as created with create_calib_obj
    roipos: Corner positions [[y,x]...]
    TODO: Save to disk in batches to remove memory bottleneck
    """
    roilist = np.zeros(len(roipos),dtype=ROIExtractor.ROIType)
    roilist['cornerpos'] = roipos
    roilist['startframe'] = roiframe
    roilist['numframes'] = 1

    with Context(ctx.smlm) as ex_ctx:
        q = None        
        numframes = 0
        for i,img in enumerate(movie):
            if i == 0:
                q = ROIExtractor(img.shape, roilist, 1, roisize, calib, ex_ctx)

            q.PushFrame(img)
            numframes += 1
 
            resultcount = q.GetResultCount()
            if resultcount > minBatchSize:
                rois,frames = q.GetResults(resultcount)
                yield rois,frames

        while not q.IsIdle():
            time.sleep(0.1)

            resultcount = q.GetResultCount()
            if resultcount > minBatchSize:
                rois,frames = q.GetResults(resultcount)
                yield rois,frames

        resultcount = q.GetResultCount()
        if resultcount>0:
            rois,frames = q.GetResults(resultcount)
            yield rois,frames


def localize_rois(rois_data_source, psf, initial_estim=None, constants=None, prog_cb=None, total=None):
    """
    Runs PSF centroid estimation on the given ROIs using the PSF.
    The PSF must be created with cuda=True
    rois_data_source can either be a file generated by detect_rois, or 
    a tuple with the ROI data (rois_info, pixels)
    """
    
    if type(rois_data_source) == str:
        iterator = load_rois_iterator(rois_data_source)
    else:
        iterator = rois_data_source

    framenums = []    
    with EstimQueue(psf,batchSize=1024) as queue, tqdm.tqdm(total=total) as pb:
        count = 0
        lastrc = 0

        def progupdate():
            nonlocal lastrc
            rc = queue.GetResultCount()
            progress = rc-lastrc
            pb.update(progress)
            lastrc = rc

            if prog_cb is None:
                return True
            return prog_cb(rc)

        for rois_info, pixels in iterator:
            roipos = np.zeros((len(rois_info), len(psf.sampleshape)))
            roipos[:,-2] = rois_info['y']
            roipos[:,-1] = rois_info['x']
            
            framenum = rois_info['id']
            framenums.append(framenum)
            if initial_estim is not None:
                initial = initial_estim[count:count+len(pixels)]
            else:
                initial = None
                
            if constants is not None:
                batch_constants = constants[count:count+len(pixels)]
            else:
                batch_constants = None
                
            queue.Schedule(pixels, ids=np.arange(count,count+len(pixels)), roipos=roipos, initial=initial, constants=batch_constants)
            count += len(pixels)

            if not progupdate():
                return None
            
        queue.Flush()
        while not queue.IsIdle():
            time.sleep(0.05)
            progupdate()
            
        progupdate()

        r = queue.GetResults(getSampleData=False)
        r.SortByID(isUnique=True) # reorder back to original ROI order
        r.ids = np.concatenate(framenums)

    return r

def create_calib_obj(gain,offset,imgshape,ctx):
    if type(gain)==str:
        print(f'estimating gain from light {gain} and dark {offset} frames')
        light = tifffile.imread(gain)
        offset = tifffile.imread(offset)
        
        if not np.array_equal(imgshape, light.shape[1:]):
            raise ValueError(f'Camera light frames calibration data ({light.shape[1:]}) does not match with expected image size ({imgshape})')
            
        if not np.array_equal(imgshape, offset.shape[1:]):
            raise ValueError(f'Camera offset calibration ({offset.shape[1:]}) does not match with expected image size ({imgshape})')

        offset = np.mean(offset,0)
        sig = light-offset
        v = np.var(sig, 0)
        m = np.mean(sig,0)
        
        gain = m/v
        gain[gain==0] = np.mean(gain)
        print(f'mean camera gain: {np.mean(gain):.2f} ADU/photons offset: {np.mean(offset):.2f}',flush=True)
        
    if type(offset)==str:
        print(f'using mean values from {offset} as camera offset',flush=True)
        offset=tiff.get_tiff_mean(offset)

        if type(gain)!=str:
            gain = np.ones(imgshape)*gain
    
    if( type(offset)==np.ndarray):
        calib = GainOffsetImage_Calib(gain, offset, ctx)
    else:
        calib = GainOffset_Calib(gain, offset, ctx) 
    
    return calib

def _summed_movie(movie, sumframes):
    img = None
    f = 0
    for m in movie:
        if f == 0:
            img = m.copy()
        else:
            img += m

        f += 1
        if f == sumframes:
            img_ = img
            img = None
            f = 0
            yield img_
            
    


class Localizer2D:
    """
    Perform localization on a tiff with a 2D Gaussian PSF model
    """    
    def __init__(self):
        ...
        
    def process(self, tiff_fn_or_iterator, cfg, output_file=None, progress_cb=None, cache_dir=None):
                
        self.cfg=cfg
        
        roisize = cfg['roisize']
        threshold = cfg['threshold']
        gain = cfg['gain']
        offset = cfg['offset']
        startframe = cfg['startframe'] if 'startframe' in cfg else 0
        maxframes = cfg['maxframes'] if 'maxframes' in cfg else -1
        sumframes = cfg['sumframes'] if 'sumframes' in  cfg else 1
        maxChiSquare = cfg['maxchisq'] if 'maxchisq' in cfg else None
        if maxChiSquare is not None and maxChiSquare == 0:
            maxChiSquare = None
            
        sigmaFramesPerBin = cfg['sigmaframesperbin'] if 'sigmaframesperbin' in cfg else None
        spotDetectSigma = cfg['spotdetectsigma']
        
        fovOffsets = cfg['fov_offsets'] if 'fov_offsets' in cfg else None
        useTiltedBg = cfg['tilted_bg'] if 'tilted_bg' in cfg else None
        
        abort=False
        def progcb(txt,prog):
            nonlocal abort
            if progress_cb is not None:
                r = progress_cb(txt,prog)
                if not r:
                    abort=True
                return r
            return True
        
        with Context() as ctx:
            
            gaussian = GaussianPSFMethods(ctx)
            
            if type(tiff_fn_or_iterator) == str:
                movie = tiff.tiff_read_file(tiff_fn_or_iterator, startframe, maxframes, progress_cb)
                if sumframes > 1:
                    movie = _summed_movie(movie, sumframes)
                    offset *= sumframes
            else:
                movie = tiff_fn_or_iterator

            if output_file is not None:
                rois_output_fn = os.path.splitext(output_file)[0]+"_rois.npy"
            else:
                rois_output_fn = os.path.splitext(tiff_fn_or_iterator)[0]+"_rois.npy"

            if cache_dir is not None:
                rois_output_fn = cache_dir + os.path.split(os.path.splitext(output_file)[0]+"_rois.npy")[1]
            self.rois_output_fn = rois_output_fn
                
            first_image, movie = peek_first(movie)
            imgshape = first_image.shape
                
            spotDetector = spotdetect.SpotDetector(spotDetectSigma, roisize, threshold)
            calib = create_calib_obj(gain,offset,imgshape,ctx)
            
            numrois,_ = detect_spots(spotDetector, calib, movie, 1, rois_output_fn, batch_size=20000, ctx=ctx)
            
            if numrois == 0:
                raise ValueError('No spots found')

            psf = gaussian.CreatePSF_XYIBg(roisize, spotDetectSigma, True)
            prog_cb = lambda cur: progress_cb(f'Fitting 2D Gaussian with approx. PSF sigma. ({cur}/{numrois})', cur/numrois)
            if progress_cb is None:
                prog_cb = None

            qr = localize_rois(rois_output_fn, psf, prog_cb=prog_cb, total=numrois)
            if qr is None:
                return
            framenum = qr.ids
    
            # Re-estimate including sigma (x,y) fits
            estim = np.zeros((len(qr.estim), 6))
            estim[:,:4] = qr.estim
            estim[:,4:] = spotDetectSigma
            psf_sigma = gaussian.CreatePSF_XYIBgSigmaXY(roisize, spotDetectSigma, True)
            prog_cb = lambda cur: progress_cb(f'Fitting 2D Gaussian including PSF sigma. ({cur}/{numrois})', cur/numrois)
            if progress_cb is None:
                prog_cb = None
            qr_sigma = localize_rois(rois_output_fn, psf_sigma, initial_estim=estim, prog_cb=prog_cb, total=numrois)
            if qr_sigma is None:
                return
    
            # Estimate per-frame sigma and interpolate using splines to account for Z drift 
            ds = Dataset.fromQueueResults(qr_sigma, imgshape)
            self.ds_sigma_fits = ds
            
            if len(ds) == 0:
                raise ValueError('PSF Sigma fits failed')
                            
            numframes = np.max(framenum)+1
            ds.data.frame = np.maximum((ds.data.frame / sigmaFramesPerBin - 0.5).astype(np.int32),0)
            frames = ds.indicesPerFrame()
            self.medianSigma = np.array([np.median(ds.data.estim.sigma[idx],0) for idx in frames])
            self.sigma_t = (0.5+np.arange(len(frames))) * sigmaFramesPerBin
            
            #self.medianSigma = [self.medianSigma[0], *self.medianSigma, self.medianSigma[-1]]
                
            self.sigma_t[0] = 0
            self.sigma_t[-1] = (len(frames)-1) * sigmaFramesPerBin
            spl_x = InterpolatedUnivariateSpline(self.sigma_t, self.medianSigma[:,0], k=2)
            spl_y = InterpolatedUnivariateSpline(self.sigma_t, self.medianSigma[:,1], k=2)
            
            self.sigma = np.zeros((numframes,2))
            self.sigma[:,0] = spl_x(np.arange(numframes))
            self.sigma[:,1] = spl_y(np.arange(numframes))
                                
            # Re-estimate using known sigmas
            psf = gaussian.CreatePSF_XYIBg(roisize, sigma=None, cuda=True)
            roi_sigmas = self.sigma[framenum]
            prog_cb = lambda cur: progress_cb(f'Fitting 2D Gaussian using interpolated PSF sigma. ({cur}/{numrois})', cur/numrois)
            if progress_cb is None:
                prog_cb = None
            r = localize_rois(rois_output_fn, psf, constants=roi_sigmas, prog_cb=prog_cb, total=numrois)
            if r is None:
                return

            ds = Dataset.fromQueueResults(r, imgshape, config=cfg, sigma=self.sigma)
            if type(tiff_fn_or_iterator) == str:
                ds['imagefile'] = tiff_fn_or_iterator
                    
            print('Filtering hitting ROI border')
            borderxy = 2.5
            lpos = ds.local_pos
            ds.filter((lpos[:,0] > borderxy) & (lpos[:,1] > borderxy) & 
                      (lpos[:,0] < roisize-borderxy-1) & (lpos[:,1] < roisize-borderxy-1))

            self.unfilteredChisq = ds.chisq*1
            if maxChiSquare is not None:
                print(f"Filtering on chi-square at threshold {maxChiSquare}*{roisize*roisize}...")
                self.chiSqThreshold = maxChiSquare*roisize**2
                ds.filter(ds.chisq<self.chiSqThreshold)
            else:
                self.chiSqThreshold = None
                   
            nframes = ds.numFrames
            print(f"Num spots: {len(ds)}. {len(r.estim) / nframes} spots/frame.")
            
            if output_file is not None:
                ds.save(output_file)

            self.result=ds
            ds.config['locs_path'] = output_file
            return ds
        
    def plotSigmaFits(self):
        plt.figure()
        plt.hist(self.result.data.estim.sigma.T, bins=100,label=['Sigma X', 'Sigma Y'])
        plt.legend()
        
    def plotChiSquare(self):
        plt.figure()
        m = np.median(self.unfilteredChisq) * 4
        plt.hist(self.unfilteredChisq, bins=100, range=[0,m], label='Chi-square values')
        if self.chiSqThreshold is not None:
            plt.axvline(self.chiSqThreshold, label='Threshold')
        plt.legend()
        plt.xlabel('Chi-square value (sum((data-model)^2/model)')

    def plotSigmaTimeSeries(self, **figargs):
        sigmaFramesPerBin = self.cfg['sigmaframesperbin']
        plt.figure(**figargs)
        plt.plot(self.sigma_t, self.medianSigma[:,0],'o')
        plt.plot(self.sigma_t, self.medianSigma[:,1],'o')
        plt.plot(self.sigma[:,0], label='Sigma X')
        plt.plot(self.sigma[:,1], label='Sigma Y')
        plt.xlabel('Frames')
        plt.ylabel('Gaussian PSF Sigma [pixels]')
        plt.legend()
        plt.title(f'PSF Sigma vs time (using {sigmaFramesPerBin} frames per bin)')
        plt.savefig(self._figpath() + "/sigma.png")
        plt.savefig(self._figpath() + "/sigma.svg")

    def _figpath(self):
        return os.path.split(self.result['locs_path'])[0]

    def plotIntensityHistogram(self):
        plt.figure()
        ph = self.result.photons
        plt.hist(self.result.photons, range=[0, np.median(ph)*4], bins=100)
        plt.xlabel('Intensity [photons]')
        plt.title('Emitter intensities')
        plt.savefig(self._figpath() + "/intensities.png")
        
        