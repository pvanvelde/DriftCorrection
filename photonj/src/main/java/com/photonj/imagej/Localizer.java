package com.photonj.imagej;

import java.awt.geom.Rectangle2D;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.Collectors;

import javax.swing.SwingUtilities;

import com.photonj.binding.Binding;
import com.photonj.binding.SpotDetectionQueue;
import com.photonj.binding.Vector3;
import com.photonj.binding.ZStack;
import com.photonj.binding.Binding.Estimator;
import com.photonj.binding.Binding.EstimatorQueue;
import com.photonj.binding.Binding.ICameraCalibration;
import com.photonj.binding.Binding.ISpotDetector;
import com.photonj.binding.Binding.ROI;
import com.photonj.binding.Binding.Spot;
import com.photonj.binding.Binding.SpotDetectParams;
import com.photonj.binding.Binding.Estimator.ChiSquareAndCRLB;
import com.photonj.binding.Binding.NativeAPI.ROIInfo;

import com.sun.jna.Pointer;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Plot;

public class Localizer {
	
	Binding binding;
	Estimator estimator;
	SpotDetectParams sdparams;
	ZStack detectionPSF;
	
	public int startFrame=0, endFrame=-1, numcoords;

	Localizer(Binding binding, Estimator estim, SpotDetectParams sdparams, ZStack detectionPSF)
	{
		this.binding=binding;
		this.estimator = estim;
		this.sdparams=sdparams;
		this.detectionPSF=detectionPSF;
		numcoords =estim.numParams()-2;
	}
	
	public void close() {
		if (estimator!=null) {
			estimator.free();
		}
	}
	
	public static Localizer Gaussian2DPSFLocalizer(int roisize, float sigmaX, float sigmaY, boolean fitSigma,  SpotDetectParams sdparams,  Binding binding)
	{
		Pointer inst;
		if (fitSigma) {
			inst = binding.getAPI().Gauss2D_CreatePSF_XYIBgSigmaXY(roisize, sigmaX, sigmaY, true, binding.getContext());
		} else {
			inst = binding.getAPI().Gauss2D_CreatePSF_XYIBg(roisize, sigmaX, sigmaY, true, binding.getContext());
		}

		Estimator estim = binding.new Estimator(inst);
		float[] params=new float[] { roisize*0.5f, roisize*0.5f, 1.0f, 0.0f, sigmaX, sigmaY };
		float[] img = estim.expectedValue(params, 1);

		ZStack psf = new ZStack(img, 
				new int[] { 1, estim.roisize(), estim.roisize()}, 
				new float[] { 0.0f, 0.0f }
			);
				
		return new Localizer(binding, estim, sdparams, psf);
	}
	
	public static Localizer CubicSplinePSFLocalizer(int roisize, String psf_file, int numZSlices, SpotDetectParams sdparams, Binding binding) {
		if (psf_file == null || !new File(psf_file).canRead())
			throw new RuntimeException("Invalid PSF file given: " + psf_file);
		
		Estimator estim = binding.createCubicSplinePSF(roisize, psf_file);

		ZStack detectionSlices = ZStack.fromPSF(estim, numZSlices);
		return new Localizer(binding, estim, sdparams, detectionSlices);
	}
	
	
	SpotDetectionQueue createSDQueue(int w, int h, ICameraCalibration cameraCalib) {
		ISpotDetector spotDetector = createSpotDetector(w, h);
		
		return new SpotDetectionQueue(spotDetector, cameraCalib, estimator.roisize(), w, h, binding) {
			
			int rejected=0;
			
			@Override
			public int minBatchSize() {
				return 16*1024;
			}
			
			float[] computeInitialValues(ROIInfo[] rois) {
				float[] initial=new float[estimator.numParams()*rois.length];
				
				int roisize = estimator.roisize();
				int np = estimator.numParams(), nc=np-2;
				for (int i=0;i<rois.length;i++) {
					initial[i*np+0] = roisize*0.5f;
					initial[i*np+1] = roisize*0.5f;
					initial[i*np+0] = detectionPSF.sliceToZPos(rois[i].z);
					initial[i*np+nc] = rois[i].score; // I
					initial[i*np+nc+1] = 0.0f; // bg
				}
				
				return initial;
			}
		
			@Override
			public void processROIs(float[] data, ROIInfo[] rois) {
				int NP=estimator.numParams();
				
				// build some initial values 
				float[] initial = computeInitialValues(rois);
				
				float[] r = estimator.estimate(data, rois.length, initial);
				float[] ev = estimator.expectedValue(r, rois.length);
				int smpcount=estimator.getSampleCount();
				
				ChiSquareAndCRLB cc = estimator.chiSquareAndCRLB(r, data, rois.length);
						
				int I = NP==5 ? 3 : 2;
				for(int i=0;i<rois.length;i++) {
					
					float x=r[NP*i+0];
					float y=r[NP*i+1];
					if(x==estimator.minValues[0] || y==estimator.minValues[1] || 
						x==estimator.maxValues[0] || y==estimator.maxValues[1]) {
						rejected++;
						continue;
					}
					
					Spot spot = new Spot();
					float[] expval = Arrays.copyOfRange(ev, i*smpcount, (i+1)*smpcount);
					float[] roi = Arrays.copyOfRange(data,i*smpcount, (i+1)*smpcount);

					spot.chiSquare=cc.chisq[i];
					spot.x=r[NP*i+0] + rois[i].x;
					spot.x_crlb = cc.crlb[NP*i+0];
					spot.y=r[NP*i+1] + rois[i].y;
					spot.y_crlb = cc.crlb[NP*i+1];
					if (numcoords==3) {
						spot.z=r[NP*i+2];
						spot.z_crlb = cc.crlb[NP*i+2];
					}
					spot.I=r[NP*i+I];
					spot.I_crlb= cc.crlb[NP*i+I];
					spot.bg=r[NP*i+I+1];
					spot.bg_crlb = cc.crlb[NP*i+I+1];
					
					if (false) {//fitSigma) {
						spot.sigmaX = r[NP*i+4];
						spot.sigmaY = r[NP*i+5];
					}
					
					spot.frame = rois[i].frame;
					
					addResult(spot);
				}
	
				System.out.println(String.format("Fitting %d rois... Accepted: %d, Rejected: %d", rois.length, getResultCount(),rejected));
				
			}
			
			@Override
			public void close() {
				super.close();
				spotDetector.destroy();
			}
		};
	}

	public LocResults localization(ImageStack stack, ICameraCalibration cameraCalib, Vector3 scaling) {
    	int maxFramesInQueue = 30;
    	    	
		System.out.println("Frames: " + stack.getSize());
		SpotDetectionQueue q = createSDQueue(stack.getWidth(), stack.getHeight(), cameraCalib);
		
    	Runnable updateStatus = ()->{				
			int roisTotal = q.numDetectedROIs();
			int roisDone = q.getResultCount();
			
			 SwingUtilities.invokeLater(new Runnable() {
			        public void run() {
						IJ.showStatus(String.format("Processing ROIs %d/%d",roisDone, roisTotal));
			        }
			    });
    	};

		int endFrame=stack.getSize();
		if(this.endFrame>=0) 
			endFrame=Math.min(endFrame, this.endFrame);
		startFrame = Math.min(startFrame, endFrame);
		startFrame= Math.max(0, startFrame);
		int numFrames=endFrame-startFrame;
		for (int i=startFrame;i<endFrame;i++) {
			while (q.getQueueLength() > maxFramesInQueue || q.getROIQueueLen() > q.minBatchSize() * 4) {
				try {
					Thread.sleep(10);
				} catch (InterruptedException e) {
				}
			}
			
			if (i % 50 == 0)
				updateStatus.run();

			Object pixels = stack.getPixels(1+i);// one-based indexing for some reason
			short[] pixels16 = (short[])pixels;
			q.pushImage(pixels16);
			
			int nf = q.numFinishedFrames();
	    	IJ.showProgress(nf, numFrames);
		}

		while (true) {
			int nf = q.numFinishedFrames();
	    	IJ.showProgress(nf, numFrames);
			if (nf < numFrames) {
				try {
					Thread.sleep(50);
				} catch (InterruptedException e) {
				}
				updateStatus.run();
			} else break;
		}
		
		q.flushROIQueue();
		
		while (!q.isFinished()) {
			try {
				Thread.sleep(50);
			} catch (InterruptedException e) {
			}
			updateStatus.run();
		}
		
		String msg = String.format("Fitted %d spots.", q.getResultCount() );
		IJ.showStatus(msg);
		System.out.println(msg);
		
		ArrayList<Spot> results;
		
		results = q.finishROIProcessing();
				
		if (results.size()>0)
			showChiSquarePlot(results);
				
		q.close();
		
		for (Spot s : results) {
			s.applyScaling(scaling);
		}
		
		if (results.isEmpty()) {
			IJ.showMessage("No spots detected");

			return null;
		}
		
		return new LocResults(results, stack.getWidth(), stack.getHeight(), scaling);
    }
    



	protected void showChiSquarePlot(ArrayList<Spot> spots) {
    	double[] chisq=new double[spots.size()];
    	for (int i=0;i<spots.size();i++)
    		chisq[i]=spots.get(i).chiSquare;
    	
    	int roisize=estimator.roisize();
    	int smpcount=roisize*roisize;
		Plot p = new Plot(String.format("Chi-Square model fit. Approx. expected value: %d", smpcount), "Chi-Square score", "Counts");
		p.addHistogram(chisq);
		//p.addLegend("X\nY");
		p.show();
	}


	public ROI[] detectSpots(ICameraCalibration cameraCalibration, ImagePlus img, int currentSlice) {
    	short[] pixels = (short[])img.getStack().getPixels(currentSlice);
    	Binding.ISpotDetector spotDetectorType = createSpotDetector(img.getWidth(), img.getHeight());
		Binding.ROI[] rois = binding.detectSpots(pixels, img.getWidth(), img.getHeight(), cameraCalibration, spotDetectorType );
		
		spotDetectorType.destroy();
		return rois;
	}

	private ISpotDetector createSpotDetector(int width, int height) {
		return binding.createPSFCorrelationSpotDetector(this.sdparams, this.detectionPSF);
	}
    

}
