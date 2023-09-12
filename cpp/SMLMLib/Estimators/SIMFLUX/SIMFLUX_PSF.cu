// SIMFLUX Estimator model
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "simflux/SIMFLUX.h"
#include "simflux/ExcitationModel.h"
#include "Estimators/Estimator.h"
#include "CudaUtils.h"
#include "Estimators/EstimatorImpl.h"
#include "SIMFLUX_PSF.h"
#include "simflux/SIMFLUX_Models.h"
#include "Estimators/Gaussian/GaussianPSFModels.h"
#include "Estimators/SIMFLUX/SIMFLUX_PSF.h"

std::vector<int> prependInt(std::vector<int> v, int a) {
	v.insert(v.begin(), a);
	return v;
}



class SIMFLUX_Gauss2D_CUDA_PSF : public cuEstimator 
{
public:
	typedef Gauss2D_Params Params;
	typedef Int3 SampleIndex;
	typedef SIMFLUX_Gauss2D_Model TModel;

	float2 sigma;
	int numframes, roisize;
	int numPatterns;

	SIMFLUX_Gauss2D_CUDA_PSF(int roisize, int numframes, int numPatterns, float2 sigma) :
		cuEstimator({ numframes,roisize,roisize },  6 * numPatterns, 0, Gauss2D_Model_XYIBg::ParamFormat(), 
			param_limits_t{
				std::vector<float>{Gauss2D_Border, Gauss2D_Border, 10.0f, 1.0f},
				std::vector<float>{roisize - 1 - Gauss2D_Border, roisize - 1 - Gauss2D_Border, 1e9f, 1e9f}
			}),
		roisize(roisize), 
		numframes(numframes), 
		sigma(sigma), 
		numPatterns(numPatterns)
	{}

	void ExpectedValue(float* d_image, const float* d_params, const float *d_const, const int* d_roipos, int numspots, cudaStream_t stream) override
	{
		const Params* theta = (const Params*)d_params;
		float2 sigma = this->sigma;
		int roisize = this->roisize;
		int numframes = this->numframes;
		int sc = SampleCount();
		const SIMFLUX_Modulation* mod = (const SIMFLUX_Modulation*)d_const;
		int numPatterns = this->numPatterns;
		int numconst = this->NumConstants();
		auto roipos = (const typename TModel::TSampleIndex *)d_roipos;
		LaunchKernel(numspots, [=]__device__(int i) {
			SineWaveExcitation exc (numPatterns, &mod[numPatterns*i]);
			TModel::Calibration calib = { exc,sigma };
			TModel model(roisize, calib, 0, numframes, numframes, roipos[i]);
			ComputeExpectedValue(theta[i], model, &d_image[sc*i]);
		}, 0, stream);
	}

	void Derivatives(float* d_deriv, float *d_expectedvalue, const float* d_params, const float *d_const, const int* d_roipos, int numspots, cudaStream_t stream) override
	{
		const Params* theta = (const Params*)d_params;
		const SIMFLUX_Modulation* mod = (const SIMFLUX_Modulation*)d_const;
		float2 sigma = this->sigma;
		int numPatterns = this->numPatterns;

		int smpcount = SampleCount();
		int roisize = this->roisize;
		int numframes = this->numframes;
		int numconst = this->NumConstants();
		int K = NumParams();
		auto roipos = (const typename TModel::TSampleIndex *)d_roipos;

		LaunchKernel(numspots, [=]__device__(int i) {
			SineWaveExcitation exc(numPatterns, &mod[numPatterns * i]);
			TModel::Calibration calib = { exc,sigma };
			TModel model(roisize, calib, 0, numframes, numframes, roipos[i]);
			ComputeDerivatives(theta[i], model, &d_deriv[i*smpcount*K], &d_expectedvalue[i*smpcount]);
		}, 0, stream);
	}

	// d_sample[numspots, SampleCount()], d_params[numspots, numparams], d_initial[numspots, NumParams()], d_params[numspots, NumParams()], d_iterations[numspots]
	void Estimate(const float* d_sample, const float *d_const, const int* d_roipos, const float* d_initial, float* d_params, float* d_diag,
		int *iterations,int numspots, float * d_trace, int traceBufLen, cudaStream_t stream)  override
	{
		int numPatterns = this->numPatterns;
		Params* theta = (Params*)d_params;
		Params* trace = (Params*)d_trace;
		const SIMFLUX_Modulation* mod = (const SIMFLUX_Modulation*)d_const;
		float2 sigma = this->sigma;
		int smpcount = SampleCount();
		int roisize = this->roisize;
		int numframes = this->numframes;
		auto roipos = (const typename TModel::TSampleIndex *)d_roipos;

		if (!d_initial)
			return;

		LevMarSettings<4> lm_settings(limits[0], limits[1], lm_step_coeff, lm_params.iterations, lm_params.normalizeWeights);
		Params* initial = (Params*)d_initial;

		LaunchKernel(numspots, [=]__device__(int i) {
			SineWaveExcitation exc(numPatterns, &mod[numPatterns * i]);
			TModel::Calibration calib = { exc,sigma };
			TModel model(roisize, calib, 0, numframes, numframes, roipos[i]);
			const float* smp = &d_sample[i*smpcount];

			auto r = LevMarOptimize(smp, initial[i], model, lm_settings, &trace[traceBufLen*i], traceBufLen);
			theta[i] = r.estimate;
			iterations[i] = r.iterations;
		}, 0, stream);
	}
};

CDLL_EXPORT Estimator* SIMFLUX_Gauss2D_CreateEstimator(int num_patterns, float sigmaX ,float sigmaY,
	int roisize, int numframes, Context* ctx)
{
	try {
		cuEstimator* cpsf;
		float2 sigma{ sigmaX,sigmaY };

		cpsf = new SIMFLUX_Gauss2D_CUDA_PSF(roisize, numframes, 
			num_patterns, sigma);

		Estimator* psf = new cuEstimatorWrapper(cpsf);
		if (ctx) psf->SetContext(ctx);
		return psf;
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}




class SIMFLUX_Gauss2DAstig_CUDA_PSF : public cuEstimator
{
public:
	typedef Vector5f Params;
	typedef Int3 SampleIndex;

	Gauss3D_Calibration psfCalib;
	int numframes, roisize;
	int numPatterns;

	SIMFLUX_Gauss2DAstig_CUDA_PSF(int roisize, int numframes, int numPatterns, Gauss3D_Calibration psfcalib) :
		cuEstimator({ numframes,roisize,roisize }, 6 * numPatterns,0, Gauss2D_Model_XYZIBg::ParamFormat(),
			param_limits_t {
				std::vector<float>{Gauss2D_Border, Gauss2D_Border, psfcalib.minz, 10.0f, 1.0f},
				std::vector<float>{roisize - 1 - Gauss2D_Border, roisize - 1 - Gauss2D_Border, psfcalib.maxz, 1e9f, 1e9f}
			}),
			roisize(roisize),
			numframes(numframes),
			psfCalib(psfcalib),
			numPatterns(numPatterns)
	{}

	void ExpectedValue(float* d_image, const float* d_params, const float* d_const, const int* d_roipos, int numspots, cudaStream_t stream) override
	{
		const Params* theta = (const Params*)d_params;
		auto psfCalib = this->psfCalib;
		int roisize = this->roisize;
		int numframes = this->numframes;
		int sc = SampleCount();
		const SIMFLUX_Modulation* mod = (const SIMFLUX_Modulation*)d_const;
		int numPatterns = this->numPatterns;
		int numconst = this->NumConstants();
		auto roipos = (const Int3*)d_roipos;
		LaunchKernel(numspots, [=]__device__(int i) {
			SineWaveExcitation exc(numPatterns, &mod[numPatterns * i]);
			SIMFLUX_Gauss2DAstig_Model model(roisize, { exc,psfCalib }, 0, numframes, numframes, roipos[i]);
			ComputeExpectedValue(theta[i], model, &d_image[sc * i]);
		}, 0, stream);
	}

	void Derivatives(float* d_deriv, float* d_expectedvalue, const float* d_params, const float* d_const, const int* d_roipos, int numspots, cudaStream_t stream) override
	{
		const Params* theta = (const Params*)d_params;
		const SIMFLUX_Modulation* mod = (const SIMFLUX_Modulation*)d_const;
		auto psfCalib = this->psfCalib;
		int numPatterns = this->numPatterns;

		int smpcount = SampleCount();
		int roisize = this->roisize;
		int numframes = this->numframes;
		int numconst = this->NumConstants();
		int K = NumParams();
		auto roipos = (const Int3*)d_roipos;

		LaunchKernel(numspots, [=]__device__(int i) {
			SineWaveExcitation exc(numPatterns, &mod[numPatterns * i]);
			SIMFLUX_Gauss2DAstig_Model model(roisize, { exc,psfCalib }, 0, numframes, numframes, roipos[i]);
			ComputeDerivatives(theta[i], model, &d_deriv[i * smpcount * K], &d_expectedvalue[i * smpcount]);
		}, 0, stream);
	}

	// d_sample[numspots, SampleCount()], d_params[numspots, numparams], d_initial[numspots, NumParams()], d_params[numspots, NumParams()], d_iterations[numspots]
	void Estimate(const float* d_sample, const float* d_const, const int* d_roipos, const float* d_initial, float* d_params, float* d_diag,
		int* iterations, int numspots, float* d_trace, int traceBufLen, cudaStream_t stream)  override
	{
		int numPatterns = this->numPatterns;
		Params* theta = (Params*)d_params;
		Params* trace = (Params*)d_trace;
		const SIMFLUX_Modulation* mod = (const SIMFLUX_Modulation*)d_const;
		auto psfCalib = this->psfCalib;
		int smpcount = SampleCount();
		int roisize = this->roisize;
		int numframes = this->numframes;
		auto roipos = (const Int3*)d_roipos;

		LevMarSettings<5> lm_settings(limits[0], limits[1], lm_step_coeff, lm_params.iterations, lm_params.normalizeWeights);

		if (!d_initial)
			return;

		Params* initial = (Params*)d_initial;
		LaunchKernel(numspots, [=]__device__(int i) {
			SineWaveExcitation exc(numPatterns, &mod[numPatterns * i]);
			SIMFLUX_Gauss2DAstig_Model model(roisize, { exc, psfCalib }, 0, numframes, numframes, roipos[i]);
			const float* smp = &d_sample[i * smpcount];

			auto r = LevMarOptimize(smp, initial[i], model, lm_settings,
				&trace[traceBufLen * i], traceBufLen);
			theta[i] = r.estimate;
			iterations[i] = r.iterations;
		}, 0, stream);
	}
};

CDLL_EXPORT Estimator* SIMFLUX_Gauss2DAstig_CreateEstimator(int num_patterns,  const Gauss3D_Calibration& calib,
	int roisize, int numframes, Context* ctx)
{
	try {
		cuEstimator* cpsf = new SIMFLUX_Gauss2DAstig_CUDA_PSF(roisize, numframes, num_patterns, calib);

		Estimator* psf = new cuEstimatorWrapper(cpsf);
		if (ctx) psf->SetContext(ctx);
		return psf;
	}
	catch (const std::runtime_error& e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}


//class SIMFLUX_Gauss2DAstig_CUDA_PSFtest : public cuEstimator
//{
//public:
//	typedef Vector5f Params;
//	typedef Int3 SampleIndex;
//
//	Gauss3D_Calibration psfCalib;
//	int numframes, roisize;
//	int numPatterns;
//
//	SIMFLUX_Gauss2DAstig_CUDA_PSFtest(int roisize, int numframes, int numPatterns, Gauss3D_Calibration psfcalib) :
//		cuEstimator({ numframes,roisize,roisize }, 6 * numPatterns, 0, Gauss2D_Model_XYZIBg::ParamFormat(),
//			param_limits_t{
//				std::vector<float>{Gauss2D_Border, Gauss2D_Border, psfcalib.minz, 10.0f, 1.0f},
//				std::vector<float>{roisize - 1 - Gauss2D_Border, roisize - 1 - Gauss2D_Border, psfcalib.maxz, 1e9f, 1e9f}
//			}),
//		roisize(roisize),
//				numframes(numframes),
//				psfCalib(psfcalib),
//				numPatterns(numPatterns)
//	{}
//
//			void ExpectedValue(float* d_image, const float* d_params, const float* d_const, const int* d_roipos, int numspots, cudaStream_t stream) override
//			{
//				const Params* theta = (const Params*)d_params;
//				auto psfCalib = this->psfCalib;
//				int roisize = this->roisize;
//				int numframes = this->numframes;
//				int sc = SampleCount();
//				const SIMFLUX_Modulation* mod = (const SIMFLUX_Modulation*)d_const;
//				int numPatterns = this->numPatterns;
//				int numconst = this->NumConstants();
//				auto roipos = (const Int3*)d_roipos;
//				LaunchKernel(numspots, [=]__device__(int i) {
//					SineWaveExcitation exc(numPatterns, &mod[numPatterns * i]);
//					SIMFLUX_Gauss2DAstig_Modeltest model(roisize, { exc,psfCalib }, 0, numframes, numframes, roipos[i]);
//					ComputeExpectedValue(theta[i], model, &d_image[sc * i]);
//				}, 0, stream);
//			}
//
//			void Derivatives(float* d_deriv, float* d_expectedvalue, const float* d_params, const float* d_const, const int* d_roipos, int numspots, cudaStream_t stream) override
//			{
//				const Params* theta = (const Params*)d_params;
//				const SIMFLUX_Modulation* mod = (const SIMFLUX_Modulation*)d_const;
//				auto psfCalib = this->psfCalib;
//				int numPatterns = this->numPatterns;
//
//				int smpcount = SampleCount();
//				int roisize = this->roisize;
//				int numframes = this->numframes;
//				int numconst = this->NumConstants();
//				int K = NumParams();
//				auto roipos = (const Int3*)d_roipos;
//
//				LaunchKernel(numspots, [=]__device__(int i) {
//					SineWaveExcitation exc(numPatterns, &mod[numPatterns * i]);
//					SIMFLUX_Gauss2DAstig_Modeltest model(roisize, { exc,psfCalib }, 0, numframes, numframes, roipos[i]);
//					ComputeDerivatives(theta[i], model, &d_deriv[i * smpcount * K], &d_expectedvalue[i * smpcount]);
//				}, 0, stream);
//			}
//
//			// d_sample[numspots, SampleCount()], d_params[numspots, numparams], d_initial[numspots, NumParams()], d_params[numspots, NumParams()], d_iterations[numspots]
//			void Estimate(const float* d_sample, const float* d_const, const int* d_roipos, const float* d_initial, float* d_params, float* d_diag,
//				int* iterations, int numspots, float* d_trace, int traceBufLen, cudaStream_t stream)  override
//			{
//				int numPatterns = this->numPatterns;
//				Params* theta = (Params*)d_params;
//				Params* trace = (Params*)d_trace;
//				const SIMFLUX_Modulation* mod = (const SIMFLUX_Modulation*)d_const;
//				auto psfCalib = this->psfCalib;
//				int smpcount = SampleCount();
//				int roisize = this->roisize;
//				int numframes = this->numframes;
//				auto roipos = (const Int3*)d_roipos;
//
//				LevMarSettings<5> lm_settings(limits[0], limits[1], lm_step_coeff, lm_params.iterations, lm_params.normalizeWeights);
//
//				if (!d_initial)
//					return;
//
//				Params* initial = (Params*)d_initial;
//				LaunchKernel(numspots, [=]__device__(int i) {
//					SineWaveExcitation exc(numPatterns, &mod[numPatterns * i]);
//					SIMFLUX_Gauss2DAstig_Modeltest model(roisize, { exc, psfCalib }, 0, numframes, numframes, roipos[i]);
//					const float* smp = &d_sample[i * smpcount];
//
//					auto r = LevMarOptimize(smp, initial[i], model, lm_settings,
//						&trace[traceBufLen * i], traceBufLen);
//					theta[i] = r.estimate;
//					iterations[i] = r.iterations;
//				}, 0, stream);
//			}
//};
//
//
//
//CDLL_EXPORT Estimator* SIMFLUX_Gauss2DAstig_CreateEstimatortest(int num_patterns, const Gauss3D_Calibration& calib,
//	int roisize, int numframes, Context* ctx)
//{
//	try {
//		cuEstimator* cpsf = new SIMFLUX_Gauss2DAstig_CUDA_PSFtest(roisize, numframes, num_patterns, calib);
//
//		Estimator* psf = new cuEstimatorWrapper(cpsf);
//		if (ctx) psf->SetContext(ctx);
//		return psf;
//	}
//	catch (const std::runtime_error& e) {
//		DebugPrintf("%s\n", e.what());
//		return 0;
//	}
//}
