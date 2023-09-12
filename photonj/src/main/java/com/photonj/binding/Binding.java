package com.photonj.binding;


import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.sun.jna.Callback;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;



public class Binding
{
	public static interface NativeAPI extends Library
	{
		public static class SpotDetectorConfig extends Structure {
			// In a nutshell:
			// detectionImage = uniform1 - uniform2
			// Selected spot locations = max(detectionImage, maxFilterSize) == detectionImage
			public int uniformFilter1Size, uniformFilter2Size, maxFilterSize;

			// Roisize is used to remove ROIs near the image border
			public int roisize;

			// Only spots where detectionImage > intensityThreshold are selected
			public float minIntensity;
			public float maxIntensity;

			public Pointer backgroundImage;

			@Override
			protected List<String> getFieldOrder() {
				return Arrays.asList(
						"uniformFilter1Size",
						"uniformFilter2Size", 
						"maxFilterSize",
						"roisize", 
						"minIntensity",
						"maxIntensity", 
						"backgroundImage");
			}			
			
			
			public SpotDetectorConfig(float sigma, int roisize, float minIntensity) {
				maxFilterSize = (int)(sigma*5);
				maxIntensity = 1e6f;
				this.minIntensity = minIntensity;
				this.roisize = roisize;
				uniformFilter1Size = (int)(sigma*2+2);
				uniformFilter2Size = 2*uniformFilter1Size;
			}
		}
		
		//sCMOS_Calibration* sCMOS_Calib_Create(int w,int h, const float* offset, const float* gain, const float *variance, Context* ctx);
		//GainOffsetCalibration* GainOffsetCalib_Create(float gain, float offset, Context* ctx);

		Pointer sCMOS_Calib_Create(int w,int h, float[] offset, float[] gain, float[] variance, Pointer ctx);
		Pointer GainOffsetCalib_Create(float gain, float offset, Pointer ctx);

		//int SpotDetector_ProcessFrame(const float* frame, int width, int height, int roisize,
		//		int maxSpots, float* spotScores,  int* spotz, Int2* cornerPos, float* rois, ISpotDetectorFactory* sdf, IDeviceImageProcessor* calib = 0);

		int SpotDetector_ProcessFrame(float[] frame, int width, int height, int roisize,
				int maxSpots, float[] spotScores,  int[] spotz, int[] cornerPosYX, float[] rois, Pointer spotDetectorType, Pointer cameraCalib);

		//void SpotDetector_DestroyFactory(ISpotDetectorFactory* factory);
		void SpotDetector_DestroyFactory(Pointer factory);
		//ISpotDetectorFactory* SpotDetector_Configure(const SpotDetectorConfig& config);
		Pointer SpotDetector_Configure(SpotDetectorConfig config);

		//void AddROIs(float* image, int width, int height, const float* rois, int numrois, int roisize, Int2* roiposYX)
		void DrawROIs(float[] image, int width, int height, float[] rois, int numrois, int roisize, int[] roiposYX);
		
		public static class CSpline_Calib extends Structure {
			@Override
			protected List<String> getFieldOrder() { 
				// TODO Auto-generated method stub
				return Arrays.asList("nz", "ny", "nx", "zmin", "zmax", "coefs");
			}

			public int nz;
			public int ny;
			public int nx;
			public float zmin;
			public float zmax;
			public Pointer coefs; // can be zero and passed as function argument
		}

		//CDLL_EXPORT void CSpline_Compute(const Int3& shape, const float* data, float* splineCoefs);
		void CSpline_Compute(int[] inputshape, float[] data, float[] splineCoeff);
		Pointer CSpline_CreatePSF(int roisize, CSpline_Calib calib, float[] coefs, boolean cuda, Pointer ctx);		

		//CDLL_EXPORT ISpotDetectorFactory* ConvolutionSpotDetector_Configure(const float* bgImage, int imageWidth, int imageHeight, 
			//									const float* psfstack, int roisize, int depth, int maxFilterSizeXY, float minPhotons, int debugMode);
		Pointer PSFCorrelationSpotDetector_Configure(Pointer bgImage, int imageWidth, int imageHeight, 
				float[] psfstack, int roisize, int depth, int maxFilterSizeXY, float minPhotons, int uniformBgFilterSize, int debugMode);


		/*CDLL_EXPORT ImageProcessor* SpotExtractionQueue_Create(int width, int height, ROIQueue* roilist,
			ISpotDetectorFactory* spotDetectorFactory, IDeviceImageProcessor* preprocessor,
			int numDetectionThreads, int sumframes, Context* ctx);
		*/
		Pointer SpotExtractionQueue_Create(int width, int height, Pointer roiQueue,
				Pointer spotDetectorFactory, Pointer cameraCalib,
				int numDetectionThreads, int sumframes, Pointer ctx);
		
		public static class EstimatorProperties extends Structure {
			public int numParams, sampleCount, numDiag, numConst, sampleIndexDims;

			@Override
			protected List<String> getFieldOrder() {
				return Arrays.asList("numParams", "sampleCount", "numDiag", "numConst", "sampleIndexDims");
			}
		}
		
		//Estimator* Gauss2D_CreatePSF_XYIBg(int roisize, float sigmaX, float sigmaY, bool cuda, Context* ctx=0);
		Pointer Gauss2D_CreatePSF_XYIBg(int roisize, float sigmaX, float sigmaY, boolean cuda, Pointer ctx);
		//Estimator* Gauss2D_CreatePSF_XYIBgSigma(int roisize, float initialSigma, bool cuda, Context* ctx=0);
		//Estimator* Gauss2D_CreatePSF_XYZIBg(int roisize, const Gauss3D_Calibration& calib, bool cuda, Context* ctx=0);
		//Estimator* Gauss2D_CreatePSF_XYIBgSigmaXY(int roisize, float initialSigmaX, float initialSigmaY, bool cuda, Context* ctx);
		Pointer Gauss2D_CreatePSF_XYIBgSigmaXY(int roisize, float initialSigmaX, float initialSigmaY, boolean cuda, Pointer ctx);
		
		//void Estimator_SampleDims(Estimator* estim, int* dims);
		void Estimator_SampleDims(Pointer estim, int[] dims);

		//void Estimator_Delete(Estimator* estim);
		void Estimator_Delete(Pointer estim);
		//const char* Estimator_ParamFormat(Estimator* estim);
		String Estimator_ParamFormat(Pointer estim);
		void Estimator_GetProperties(Pointer estim, EstimatorProperties[] props);
		//void Estimator_GetProperties(Estimator* estim, EstimatorProperties& props);

		public static class LMParams extends Structure {
			public int iterations;
			public float lambda; // levenberg-marquardt damping factor

			// Adaptive adjustment of lambda: if 0 no adjustment. 
			// If nonzero, improvement in likelihood will divide lambda by lmFactor, no improvement will multiply it
			public float factor;

			@Override
			protected List<String> getFieldOrder() {
				return Arrays.asList("iterations", "lambda", "factor");
			}
		}


		//void Estimator_SetLMParams(Estimator* estim, const LMParams&);
		void Estimator_SetLMParams(Pointer estim, LMParams[] lmp);
		//CDLL_EXPORT void Estimator_GetParamLimits(Estimator* estim, ParamLimit* limits);
		void Estimator_GetParamLimits(Pointer estim, float[] limits);
	
		//void Estimator_ComputeExpectedValue(Estimator* estim, int numspots, const float* params, const float* constants, const int* spotpos, float* ev);
		void Estimator_ComputeExpectedValue(Pointer estim, int numspots, float[] params, Pointer constants, Pointer spotpos, float[] ev);
		void Estimator_Estimate(Pointer estim, int numspots, float[] sample, Pointer constants, Pointer spotpos,
				float[] initial, float[] params, Pointer diagnostics, Pointer iterations, Pointer trace, int traceBufLen);
		//void Estimator_Estimate(Estimator* estim, int numspots, const float* sample, const float* constants, const int* spotpos,
		//	const float* initial, float* params, float* diagnostics, int* iterations, float* trace, int traceBufLen);
		//void Estimator_ChiSquareAndCRLB(Estimator* estim, const float* params, const float* sample, const float* h_const, const int* spot_pos, int numspots, float* crlb, float* chisq)
		void Estimator_ChiSquareAndCRLB(Pointer estim, float[] params, float[] sample, Pointer h_const, Pointer spot_pos, 
				int numspots, float[] crlb, float[] chisq);

		//EstimationQueue* EstimQueue_Create(Estimator* psf, int batchSize, int maxQueueLen, int numStreams, Context* ctx);
		Pointer EstimQueue_Create(Pointer estim, int batchSize, int maxQueueLen, int numStreams, Pointer ctx);
		//void EstimQueue_Delete(EstimationQueue* queue);
		void EstimQueue_Delete(Pointer q);

		int EstimQueue_GetQueueLength(Pointer q);
		
		//void EstimQueue_Schedule(EstimationQueue* q, int numspots, const int *ids, const float* h_samples, const float* h_initial,
		//	const float* h_constants, const int* h_roipos);
		void EstimQueue_Schedule(Pointer q, int numspots, int[] ids, float[] h_samples, float[] h_initial, Pointer constants, Pointer roipos);
		
		//void EstimQueue_Flush(EstimationQueue* q);
		void EstimQueue_Flush(Pointer q);
		//bool EstimQueue_IsIdle(EstimationQueue* q);
		boolean EstimQueue_IsIdle(Pointer q);
		
		//int EstimQueue_GetResultCount(EstimationQueue* q);
		int EstimQueue_GetResultCount(Pointer q);
		

		//@FieldOrder({"id", "score", "x", "y", "z"})
		public static class EstimatorResult extends Structure {
			public int id;
			public float chisq;
			public int iterations;

			public EstimatorResult() {}
			@Override
			protected List<String> getFieldOrder() { 
				// TODO Auto-generated method stub
				return Arrays.asList("id", "chisq", "iterations");
			}
		}
		
		// Returns the number of actual returned localizations. 
		// Results are removed from the queue after copyInProgress to the provided memory
		//int EstimQueue_GetResults(EstimationQueue* q, int maxresults, float* estim, float* diag,
		// float* crlb, int* roipos, float* samples, EstimatorResult* result);
		int EstimQueue_GetResults(Pointer q, int maxresults, float[] estim, float[] diag, float[] crlb, int[] roipos, float[] samples, EstimatorResult[] result);
		
		//int EstimQueue_GetQueueLength(EstimationQueue* q);

		
		
		
		//@FieldOrder({"id", "score", "x", "y", "z"})
		public static class ROIInfo extends Structure {
			public int frame;
			public float score;
			public int x,y,z;

			public ROIInfo() {}
			@Override
			protected List<String> getFieldOrder() { 
				// TODO Auto-generated method stub
				return Arrays.asList("frame", "score", "x", "y", "z");
			}
		}
		
//		CDLL_EXPORT ROIQueue* RQ_Create(const Int3& shape, Context*ctx);
		Pointer RQ_Create(int[] shape, Pointer ctx);
		//CDLL_EXPORT void RQ_Pop(ROIQueue*q, int count, ROIInfo* rois, float* data);
		void RQ_Pop(Pointer q, int count, ROIInfo[] rois, float[] data);
		//CDLL_EXPORT int RQ_Length(ROIQueue* q);
		int RQ_Length(Pointer q);
		//CDLL_EXPORT void RQ_Push(ROIQueue* q, int count ,const ROIInfo* info, const float* data);
		//void RQ_Push(ROIQueue* q, int count ,const ROIInfo* info, const float* data);
		//CDLL_EXPORT void RQ_SmpShape(ROIQueue* q, Int3* shape);
		void RQ_SmpShape(Pointer rq, int[] shape);
		
		//void ImgProc_AddFrame(ImageProcessor* q, uint16_t* data);
		void ImgProc_AddFrame(Pointer q, short[] data);
		void ImgProc_Destroy(Pointer q);
		//int ImgProc_GetQueueLength(ImageProcessor* p);
		int ImgProc_GetQueueLength(Pointer p);
		//int ImgProc_ReadFrame(ImageProcessor* q, float* image, float* processed);
		//int ImgProc_ReadFrame(Pointer q, float* image, float* processed);
		//int ImgProc_NumFinishedFrames(ImageProcessor * q);
		int ImgProc_NumFinishedFrames(Pointer q);
		//bool ImgProc_IsIdle(ImageProcessor* q);
		boolean ImgProc_IsIdle(Pointer q);

		Pointer Context_Create();
		void Context_Destroy(Pointer ctx);
		
		public interface DebugPrintCallback extends Callback {
			int callback(String value);
		}
		
		//void SetDebugPrintCallback(void(*cb)(const char* msg));
		void SetDebugPrintCallback(DebugPrintCallback cb);

	}
   
   	NativeAPI api;
   	Pointer context;
   	NativeAPI.DebugPrintCallback debugPrintCallback;
   	
   	public Pointer getContext() {
   		return context;
   	}

   	public Binding(String path)
   	{
   		api = (NativeAPI)Native.loadLibrary(path, NativeAPI.class);
   		
   		debugPrintCallback = new NativeAPI.DebugPrintCallback() {
			@Override
			public int callback(String value) {
				System.out.print(value);
				return 0;
			}
		};
   		api.SetDebugPrintCallback(debugPrintCallback);

   		context = api.Context_Create();
	}
   	
   	
	public static interface ImageQueue
	{
		void pushImage(short[] image);
		boolean isIdle();
		int numFinishedFrames();
		int getQueueLength();
	}


	
	public class Estimator {
		Pointer ptr;
		NativeAPI.EstimatorProperties props;
		String paramFormat;
		ArrayList<String> paramNames=new ArrayList<String>();
		int[] sampleDims;
		
		public float[] minValues, maxValues;
		
		public Estimator(Pointer ptr) {
			this.ptr=ptr;			
			NativeAPI.EstimatorProperties[] props=new NativeAPI.EstimatorProperties[1];
			
			api.Estimator_GetProperties(ptr, props);
			this.props=props[0];
			
			sampleDims=new int[props[0].sampleIndexDims];
			api.Estimator_SampleDims(ptr, sampleDims);
			
			paramFormat = api.Estimator_ParamFormat(ptr);
			for(String s : paramFormat.split(","))
				paramNames.add(s.trim());
			
			float[] limits=new float[2*numParams()];
			api.Estimator_GetParamLimits(ptr, limits);
			
			minValues=new float[numParams()];
			maxValues=new float[numParams()];
			for (int i=0;i<numParams();i++)
			{
				minValues[i]=limits[i*2+0];
				maxValues[i]=limits[i*2+1];
			}
		}
		
		public int roisize() {
			return sampleDims[sampleDims.length-1];
		}
		
		public int getSampleCount() {
			return props.sampleCount;
		}
		
		public int numParams() {
			return props.numParams;
		}
		
		public float[] estimate(float[] data, int numspots, float[] initial)
		{
			float[] params=new float[props.numParams*numspots];
			assert(data.length==props.sampleCount * numspots);
			
			assert(initial == null || initial.length == params.length);
			
			api.Estimator_Estimate(ptr, numspots, data, Pointer.NULL, Pointer.NULL, initial, params, Pointer.NULL, Pointer.NULL, Pointer.NULL, 0);
			return params;
		}
		
		public float[] expectedValue(float[] params, int numspots)
		{
			float[] ev = new float[props.sampleCount*numspots];
			api.Estimator_ComputeExpectedValue(ptr, numspots, params, Pointer.NULL, Pointer.NULL, ev);
			return ev;
		}
		
		public class ChiSquareAndCRLB {
			public float[] crlb, chisq;
		}
		
		public ChiSquareAndCRLB chiSquareAndCRLB(float[] params, float[] sample, int numspots) {
			ChiSquareAndCRLB r = new ChiSquareAndCRLB();
			r.crlb=new float[props.numParams*numspots];
			r.chisq = new float[numspots];
			api.Estimator_ChiSquareAndCRLB(ptr, params, sample, Pointer.NULL, Pointer.NULL , numspots, r.crlb, r.chisq);
			return r;
		}
		
		public void free() {
			if (ptr != null) {
				api.Estimator_Delete(ptr);
				ptr=null;
			}
		}
	}
	
	public class EstimatorQueue {
		Pointer q;
		Estimator estim;
		
		public int numStreams() {
			return 3;
		}
		
		
		public EstimatorQueue(Estimator estim, int batchSize) {
			int maxQueueLenInBatches=5;
			this.estim=estim;
			q = api.EstimQueue_Create(estim.ptr, batchSize, maxQueueLenInBatches, numStreams(), context);
		}
		
		public void schedule(int numspots, int[] ids, float[] data, float[] initial) {
			
			assert(data.length == numspots * estim.getSampleCount());
			assert(initial == null || initial.length == numspots * estim.numParams());
			
			api.EstimQueue_Schedule(q, numspots, ids, data, initial, Pointer.NULL, Pointer.NULL);
		}
		
		public int getQueueLength() {
			return api.EstimQueue_GetQueueLength(q);
		}
		
		
		public int getResultCount() {
			return api.EstimQueue_GetResultCount(q);
		}
		
		public class Results {
			public float[] chisq;
			public int[] id;
			public float[] estimates;
			public float[] crlb;
		}

		public Results fetchResults(int maxResults) {
			
			NativeAPI.EstimatorResult[] er= new NativeAPI.EstimatorResult[maxResults];
			
			Results r  = new Results();
			r.crlb = new float[maxResults*estim.numParams()];
			r.estimates = new float[maxResults*estim.numParams()];
			
			int count = api.EstimQueue_GetResults(q, maxResults, r.estimates, null, r.crlb, null, null, er);
			r.chisq=new float[count];
			r.id = new int[count];
			for (int i=0;i<count;i++) {
				r.chisq[i]=er[i].chisq;
				r.id[i]=er[i].id;
			}
			return r;
		}
		
		// Make sure that the last bit of data that is smaller than batchSize also gets processed.
		public void flush() {
			api.EstimQueue_Flush(q);
		}
		
		public void free() {
			if (q != null) {
				api.EstimQueue_Delete(q);
				q=null;
			}
		}
	}

	public Estimator Gaussian2DEstimator(int roisize, float sigmaX,float sigmaY) {
		Pointer p =api.Gauss2D_CreatePSF_XYIBg(roisize, sigmaX, sigmaY, true, context);
		
		/*
		NativeAPI.LMParams[] lmp = new NativeAPI.LMParams[1];
		lmp[0].iterations = 40;
		lmp[0].lambda = -1.0f;
		lmp[0].factor=0.0f;
		
		api.Estimator_SetLMParams(p, lmp);
		*/
		return new Estimator(p);
	}
	

	public static class Spot implements IHasLocation {
		public int frame;
		public float[] roi, expval;
		public float chiSquare;
		public float x, y, z, I, bg, sigmaX, sigmaY;
		public float x_crlb, y_crlb, z_crlb, I_crlb, bg_crlb;

		@Override
		public Vector3 getLocation() {
			return new Vector3(x,y,z);
		}

		public void applyScaling(Vector3 scaling) {
			x *= scaling.x;
			x_crlb *= scaling.x;
			sigmaX *= scaling.x;
			
			y *= scaling.y;
			y_crlb *= scaling.y;
			sigmaY *= scaling.y;
			
			z *= scaling.z;
			z_crlb *= scaling.z;
		}
	}
	
	public float[] RenderGauss2D(int width, int height, float[] spotX, float[]spotY, float[] spotI,float[] spotSigmaX, float[] spotSigmaY, float sigma, int roisize) {
		
		float[] image = new float[width*height];
		
		if (spotX.length==0)
			return image;
		
		int numspots=spotX.length;

		Estimator e = null;
		if (sigma > 0.0f)  {
			Pointer p =api.Gauss2D_CreatePSF_XYIBg(roisize, sigma, sigma, true, context);
			e = new Estimator(p);
		} else {
			Pointer p =api.Gauss2D_CreatePSF_XYIBgSigmaXY(roisize, 1.0f, 1.0f, true, context);
			e = new Estimator(p);
			
		}

		int roi_w = e.sampleDims[1]; // python layout (rows, cols..)
		int roi_h = e.sampleDims[0];
		int[] roipos=new int[2*numspots];

		int np = e.numParams();
		float[] params=new float[e.numParams()*numspots];
		for (int i=0;i<numspots;i++) {
			
			float sx=spotX[i];
			float sy=spotY[i];
			
			// note that AddROIs already does clipping
			int roix = (int)(sx-roi_w/2);
			int roiy = (int)(sy-roi_h/2);
			
			roipos[i*2+0] = roiy;
			roipos[i*2+1] = roix;
			
			params[i*np+0] = sx-roix;
			params[i*np+1] = sy-roiy;
			params[i*np+2] = spotI[i];
			params[i*np+3] = 0.0f;
			
			if (spotSigmaX!=null)  {
				params [i*np+4] = spotSigmaX[i];
				params [i*np+5] = spotSigmaY[i];
			}
		}
		
		float[] ev = e.expectedValue(params,numspots);
				
		api.DrawROIs(image, width, height, ev, numspots, roisize, roipos);
		
		e.free();
		
		return image;
	}

   	public void Close()
   	{
   		System.out.println("Deleting native objects");
   		
   		if (debugPrintCallback != null) {
   			api.SetDebugPrintCallback(null);
   			debugPrintCallback = null; // this allows GC collection
   		}
   		
   		if (context != null)
   			api.Context_Destroy(context);;
   		context = null;
   	}

	public NativeAPI getAPI() {
		// TODO Auto-generated method stub
		return api;
	}
	
	public static interface ICameraCalibration {
		Pointer getNativeType();
		void destroy();
	}
	
	public ICameraCalibration createCameraCalibration(float gain, float offset) {
		return new ICameraCalibration() { 
			Pointer ctx, calib;
			
			@Override
			public Pointer getNativeType() {
				if (calib == null) {
					ctx= api.Context_Create();
					calib = api.GainOffsetCalib_Create(gain, offset, ctx);
				}
				return calib;
			}

			@Override
			public void destroy() {
				api.Context_Destroy(ctx);
			}
		};
	}
	
	public static interface ISpotDetector {
		Pointer getNativeType();
		int getROISize();
		void destroy();
	}
	
	public class ROI {
		public int x,y,z;
		public float[] pixels;
	}

	public ROI[] detectSpots(short[] pixels, int w, int h, ICameraCalibration cameraCalibration, ISpotDetector spotDetector) {

		float[] frame = new float[pixels.length];
		for (int i=0;i<pixels.length;i++)
			frame[i]=pixels[i];
		
		int maxSpots=500;
		float[] spotScores=new float[maxSpots];
		int[] spotz =new int[maxSpots];
		int[] cornerPosYX = new int[maxSpots*2];
		int roisize=spotDetector.getROISize();
		int smpcount=roisize*roisize;
		float[] roidata=new float[smpcount*maxSpots];
		int numspots = api.SpotDetector_ProcessFrame(frame, w, h, roisize, maxSpots, spotScores, spotz, cornerPosYX, roidata, 
				spotDetector.getNativeType(), cameraCalibration.getNativeType());

		ROI[] rois =new ROI[numspots];
		for (int i=0;i<numspots;i++) {
			ROI roi = new ROI();
			roi.x=cornerPosYX[2*i+1];
			roi.y=cornerPosYX[2*i+0];
			roi.z=spotz[i];
			roi.pixels=Arrays.copyOfRange(roidata, smpcount*i, (i+1)*smpcount);
			rois[i]=roi;
		}
		return rois;
	}

	public ISpotDetector createUniformFilterSpotDetector(NativeAPI.SpotDetectorConfig sdcfg) {
		
		return new ISpotDetector() {
			
			Pointer nativeObj;
			
			@Override
			public int getROISize() {
				return sdcfg.roisize;
			}
			
			@Override
			public Pointer getNativeType() {
				if (nativeObj != null)
					return nativeObj;
				nativeObj = api.SpotDetector_Configure(sdcfg);
				return nativeObj;
			}

			@Override
			public void destroy() {
				api.SpotDetector_DestroyFactory(nativeObj);
				nativeObj = null;
			}
		};
	}
	

	public static class SpotDetectParams {
		public float minThreshold; 
		public int bgUniformFilter;
		public float[] bgImage;
		public int maxFilterSizeXY;
		public int imageWidth, imageHeight;
		public boolean debugMode;
	}
	
	
	public ISpotDetector createPSFCorrelationSpotDetector(SpotDetectParams cfg, ZStack psf) {
		return new ISpotDetector() {
			Pointer nativeObj;
			
			@Override
			public int getROISize() {
				return psf.shape[2];
			}
			
			@Override
			public Pointer getNativeType() {
				Pointer bgimg=null;//todo
				return api.PSFCorrelationSpotDetector_Configure(bgimg, cfg.imageWidth, cfg.imageHeight, psf.data, getROISize(), 
						psf.shape[0], cfg.maxFilterSizeXY, cfg.minThreshold, cfg.bgUniformFilter, cfg.debugMode ? 1 : 0 );
			}
			
			@Override
			public void destroy() {
				api.SpotDetector_DestroyFactory(nativeObj);
				nativeObj = null;
			}
		};
	}
	

	public Estimator createCubicSplinePSF(int roisize, String psf_file) {
		ZStack zstack;
		try {
			zstack = ZStack.readFromFile(psf_file);
		} catch (IOException e) {
			throw new RuntimeException(e.getMessage());
		}
		
		int numcoefs = 64;
		int shape[] = zstack.shape;
		for (int i=0;i<shape.length;i++)
			numcoefs*=shape[i]-3;
		float[] splineCoefs = new float[numcoefs];
		
		api.CSpline_Compute(shape, zstack.data, splineCoefs);
		
		NativeAPI.CSpline_Calib calib= new NativeAPI.CSpline_Calib();
		calib.nx=shape[2]-3;
		calib.ny=shape[1]-3;
		calib.nz=shape[0]-3;
		calib.zmax=zstack.zrange[1];
		calib.zmin=zstack.zrange[0];
		calib.coefs = Pointer.NULL;
		
		Pointer ptr = api.CSpline_CreatePSF(roisize, calib, splineCoefs, true, context);
		
		return new Estimator(ptr);
	}

}
