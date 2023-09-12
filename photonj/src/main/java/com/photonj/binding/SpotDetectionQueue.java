package com.photonj.binding;

import java.util.ArrayList;

import com.photonj.binding.Binding.ICameraCalibration;
import com.photonj.binding.Binding.ISpotDetector;
import com.photonj.binding.Binding.ImageQueue;
import com.photonj.binding.Binding.Spot;
import com.photonj.binding.Binding.NativeAPI.ROIInfo;
import com.sun.jna.Pointer;


public abstract class SpotDetectionQueue implements AutoCloseable, Binding.ImageQueue
{
	Pointer sdcontext;
	Pointer roiQueue; // deallocated by sdcontext
	Pointer detectionQueue;
	Binding api;
	
	int totalROIs=0, totalFrames=0;
	int roisize;
	int roisInProcessing=0;
	
	ImageQueue imgQueue;
	Thread sendToLocalizerThread;
	ArrayList<Spot> results;
	
	public int minBatchSize() {
		return 8*1024;
	}
		
	public int getResultCount() {
		synchronized(results) {
			return results.size();
		}
	}

	public int numDetectedROIs() {
		return totalROIs;
	}
	
	public boolean isFinished() {
		int numFinished=numFinishedFrames();
		if (numFinished < totalFrames)
			return false;

		int numROIs = api.api.RQ_Length(roiQueue);
		
		synchronized(results) {
			return numROIs == 0 && roisInProcessing == 0;
		}
	}
	
	volatile boolean stopThread=false, flush=false;
	
	protected abstract void processROIs(float[] data, ROIInfo[] rois);

	protected void addResult(Spot spot) {
		synchronized(results) {
			results.add(spot);
		}
	}
	
	public SpotDetectionQueue(ISpotDetector sd, ICameraCalibration cameraCalib, int roisize, int imgWidth, int imgHeight, Binding api )
	{
		results = new ArrayList<Spot>();
		this.api = api;
		this.roisize=roisize;
		sdcontext = api.api.Context_Create();
		
		createROIQueue(roisize);
		
		int numDetectionThreads=1;
		
		detectionQueue = api.api.SpotExtractionQueue_Create(imgWidth, imgHeight, roiQueue,
				sd.getNativeType(), cameraCalib.getNativeType(), numDetectionThreads, 1, sdcontext);
					
		sendToLocalizerThread = new Thread() {
			public void run () {
				while (!stopThread) {
					int numAvail = api.api.RQ_Length(roiQueue);
					
					if (numAvail > minBatchSize() || (numAvail>0 && flush)) {
						totalROIs += numAvail;
						
						ROIInfo[] rois= new ROIInfo[numAvail];
						float[] data=new float[roisize*roisize*numAvail];

						synchronized(results) {
							roisInProcessing+=numAvail;
						}
						
						api.api.RQ_Pop(roiQueue, numAvail, rois, data);
						
						processROIs(data, rois);

						synchronized(results) {
							roisInProcessing=0;
						}

					} else {
						try {
							Thread.sleep(10);
						} catch (InterruptedException e) {
						}
					}
				}
			}
		};
		
		sendToLocalizerThread.start();
	}
		
	public void flushROIQueue() {
		this.flush=true;
	}
	
	public ArrayList<Spot> finishROIProcessing() {
		stopThread=true;
		try {
			sendToLocalizerThread.join();
			
		} catch (InterruptedException e) {
		}
		return results;
	}
		
	void createROIQueue(int roisize)
	{
        int[] shape = new int[] { 1,roisize,roisize};
		
		roiQueue = api.api.RQ_Create(shape, sdcontext);
	}
	
	@Override
	public void pushImage(short[] image) {
		totalFrames++;
		api.api.ImgProc_AddFrame(detectionQueue, image);
	}

	@Override
	public boolean isIdle() {
		return api.api.ImgProc_IsIdle(detectionQueue);
	}

	@Override
	public int numFinishedFrames() {
		return api.api.ImgProc_NumFinishedFrames(detectionQueue);
	}

	@Override
	public int getQueueLength() {
		return api.api.ImgProc_GetQueueLength(detectionQueue);
	}
		
		
	@Override
	public void close() {
		finishROIProcessing();

		if (detectionQueue != null) {
			api.api.ImgProc_Destroy(detectionQueue);
			detectionQueue = null;
		}
		
		if (sdcontext!=null) {
			api.api.Context_Destroy(sdcontext); // this deletes roiqueue on C++ side
			sdcontext=null;
		}
	}

	public int getROIQueueLen() {
		return api.api.RQ_Length(roiQueue);
	}

	public int getROISize() {
		return roisize;
	}

}



