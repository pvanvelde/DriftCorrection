// Estimator CUDA wrapper
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "EstimatorImpl.h"
#include "CudaUtils.h"
#include <unordered_map>

// For ComputeCOM
#include "Gaussian/GaussianPSFModels.h"

CDLL_EXPORT Estimator* Estimator_WrapCUDA(cuEstimator* psf)
{
	return new cuEstimatorWrapper(psf);
}



cuEstimator::DeviceBuffers::DeviceBuffers(int smpcount, int numspots, int k)
	: numspots(numspots), 
	psf_deriv(smpcount*k*numspots), 
	psf_ev(smpcount*numspots), 
	lm_alphabeta(numspots*k*(k+1)), 
	lm_lu(numspots*k*k),
	invert_temp(numspots*(k+1))
{}

cuEstimator::DeviceBuffers::~DeviceBuffers() {}

cuEstimator::DeviceBuffers* cuEstimator::GetDeviceBuffers(cudaStream_t stream, int numspots)
{
	return LockedFunction(streamDataMutex, [&]() {
		auto it = streamData.find(stream);

		if (it != streamData.end() && it->second.numspots < numspots) {
			streamData.erase(it);
			it = streamData.end();
		}

		if (it == streamData.end())
			it = streamData.emplace(stream, DeviceBuffers(SampleCount(), numspots, NumParams())).first;
		return &it->second;
		});
}


cuEstimator::cuEstimator(const std::vector<int>& sampleSize, int numConst, int diagsize, 
	const char* paramFormat, param_limits_t limits) : 
	EstimatorBase(sampleSize, numConst, diagsize, paramFormat, limits)
{}

cuEstimator::~cuEstimator()
{}


void cuEstimator::ChiSquareAndCRLB(const float* d_params, const float* sample, const float* d_const, 
	const int* d_roipos,  float* crlb, float* chisq, int numspots, cudaStream_t stream)
{
	auto buffers = GetDeviceBuffers(stream, numspots);

	float* d_deriv = buffers->psf_deriv.data();
	float* d_ev = buffers->psf_ev.data();
	float* d_fi = buffers->lm_lu.data(); // happens to be also K*K elements
	float* d_fi_inv = buffers->lm_alphabeta.data();
	int* d_P = buffers->invert_temp.data();

	Derivatives(d_deriv, d_ev, d_params, d_const, d_roipos, numspots, stream);

	int K = NumParams();
	int smpcount = SampleCount();

	LaunchKernel(numspots, [=]__device__(int spot) {
		float* spot_crlb = &crlb[K * spot];
		float* fi = &d_fi[K*K*spot];
		for (int i = 0; i < K*K; i++)
			fi[i] = 0;

		float spot_chisq = 0.0f;
		const float *spot_deriv = &d_deriv[spot*smpcount*K];

		for (int i = 0; i < smpcount; i++) {
			float mu = d_ev[spot * smpcount + i];
			auto jacobian = [=](int j) { return spot_deriv[smpcount*j + i]; };

			mu = max(1e-8f, mu);

			if (sample) {
				float err = sample[smpcount * spot + i] - mu;
				spot_chisq += err * err / mu;
			}

			float inv_mu_c = 1.0f / mu;
			for (int i = 0; i < K; i++) {
				for (int j = i; j < K; j++) {
					const float fi_ij = jacobian(i) * jacobian(j) * inv_mu_c;
					fi[K*i + j] += fi_ij;
				}
			};
		}
		// fill below diagonal
		for (int i = 1; i < K; i++)
			for (int j = 0; j < i; j++)
				fi[K*i + j] = fi[K*j + i];

		float* fi_inv = &d_fi_inv[K * K * spot];
		InvertMatrix(K, fi, &d_P[(K + 1) * spot], fi_inv);

		for (int i = 0; i < K; i++)
			spot_crlb[i] = sqrtf(fi_inv[i * K + i]);

		chisq[spot] = spot_chisq;
	}, 0, stream);
}


cuEstimatorWrapper::cuEstimatorWrapper(cuEstimator * cudaPSF) : 
	Estimator(cudaPSF->SampleSize(), cudaPSF->NumConstants(), cudaPSF->DiagSize(), cudaPSF->ParamFormat(), cudaPSF->GetLimits()), psf(cudaPSF)
{}


cuEstimatorWrapper::~cuEstimatorWrapper()
{
	delete psf;
}

void cuEstimatorWrapper::ChiSquareAndCRLB(const float* h_params, const float* h_sample, const float* h_const,
	const int* h_roipos, float* h_crlb, float* h_chisq, int numspots)
{
	DeviceArray<float> d_chisq(numspots);
	DeviceArray<float> d_smp(numspots * SampleCount(), h_sample);
	DeviceArray<float> d_params(numspots*NumParams(), h_params);
	DeviceArray<float> d_crlb(numspots * NumParams(), h_params);
	DeviceArray<float> d_const(numspots*NumConstants(), h_const);
	DeviceArray<int> d_roipos(numspots*SampleIndexDims(), h_roipos);
	psf->ChiSquareAndCRLB(d_params.ptr(), h_sample ? d_smp.ptr() : 0, d_const.ptr(), d_roipos.ptr(), d_crlb.ptr(), d_chisq.ptr(), numspots, 0);
	
	if (h_chisq) d_chisq.CopyToHost(h_chisq);
	if (h_crlb) d_crlb.CopyToHost(h_crlb);
}


void cuEstimatorWrapper::ExpectedValue(float * h_expectedvalue, const float * h_theta, const float* h_const, const int* spot_pos, int numspots)
{
	DeviceArray<float> d_ev(numspots*SampleCount());
	DeviceArray<float> d_params(numspots*NumParams(), h_theta);
	DeviceArray<float> d_const(numspots*NumConstants(), h_const);
	DeviceArray<int> d_roipos(numspots*SampleIndexDims(), spot_pos);
	psf->ExpectedValue(d_ev.ptr(), d_params.ptr(), d_const.ptr(), d_roipos.ptr(), numspots, 0);
	d_ev.CopyToHost(h_expectedvalue);
}

void cuEstimatorWrapper::Derivatives(float * h_deriv, float * h_expectedvalue, const float * h_theta, const float* h_const, const int* spot_pos, int numspots)
{
	DeviceArray<float> d_params(numspots*NumParams(), h_theta);
	DeviceArray<float> d_ev(numspots*SampleCount());
	DeviceArray<float> d_deriv(numspots*NumParams()*psf->SampleCount());
	DeviceArray<float> d_const(numspots*NumConstants(), h_const);
	DeviceArray<int> d_roipos(numspots*SampleIndexDims(), spot_pos);
	psf->Derivatives(d_deriv.ptr(), d_ev.ptr(), d_params.ptr(), d_const.ptr(), d_roipos.ptr(), numspots, 0);
	d_ev.CopyToHost(h_expectedvalue);
	d_deriv.CopyToHost(h_deriv);
}

void cuEstimatorWrapper::Estimate(const float * h_sample, const float* h_const, const int* spot_pos, const float * h_initial, 
	float * h_theta, float * h_diag, int* h_iterations, int numspots, float* h_trace, int traceBufLen)
{
	DeviceArray<float> d_smp(numspots*SampleCount(), h_sample);
	DeviceArray<float> d_initial(h_initial ? numspots * NumParams() : 0, h_initial);
	DeviceArray<float> d_params(numspots*NumParams());
	DeviceArray<float> d_diag(numspots*DiagSize());
	DeviceArray<float> d_trace(numspots*traceBufLen*NumParams());
	DeviceArray<float> d_const(numspots*NumConstants(), h_const);
	DeviceArray<int> d_roipos(numspots*SampleIndexDims(), spot_pos);
	DeviceArray<int> d_iterations(numspots);
	psf->Estimate(d_smp.ptr(), d_const.ptr(), d_roipos.ptr(), d_initial.ptr(), d_params.ptr(), 
		d_diag.ptr(), d_iterations.ptr(), numspots, d_trace.ptr(), traceBufLen, 0);
	d_params.CopyToHost(h_theta);
	if (h_trace) d_trace.CopyToHost(h_trace);
	if (h_diag) d_diag.CopyToHost(h_diag);
	if (h_iterations) d_iterations.CopyToHost(h_iterations);
}


CenterOfMassEstimator::CenterOfMassEstimator(int roisize) : roisize(roisize),
	cuEstimator({ roisize,roisize }, 0, 0, "x,y,I,bg", param_limits_t{
			std::vector<float>({0.0f, 0.0f, 0.0f, 1.0f }),
			std::vector<float>({roisize - 1.0f, roisize - 1.0f, 1e9f,1e9f})
		})
{}

void CenterOfMassEstimator::Estimate(const float* d_sample, const float *d_const, const int* d_roipos, const float* d_initial, float* d_params, float* d_diag, int* d_iterations,
	int numspots, float * d_trace, int traceBufLen, cudaStream_t stream)
{
	int numThreads = 1024;
	int roisize = SampleSize()[0];
	Vector4f* estim = (Vector4f*)d_params;
	LaunchKernel(numspots, [=]__device__(int i) {
		const float* smp = &d_sample[i*roisize*roisize];
		auto com = ComputeCOM(smp, { roisize,roisize });
		estim[i] = { com[0],com[1],com[2],0.0f };
		d_iterations[i] = 0;
	}, 0, stream, numThreads);
}

PhasorEstimator::PhasorEstimator(int roisize) : roisize(roisize), 
	cuEstimator({ roisize,roisize }, 0, 0, "x,y,I,bg", { 
			std::vector<float>({0.0f, 0.0f, 0.0f, 1.0f }),
			std::vector<float>({roisize - 1.0f, roisize - 1.0f, 1e9f,1e9f})
	})
{}

void PhasorEstimator::Estimate(const float* d_sample, const float *d_const, const int* d_roipos, const float* d_initial, 
	float* d_params, float* d_diag, int* d_iterations, int numspots, float * d_trace, int traceBufLen, cudaStream_t stream)
{
	int roisize = SampleSize()[0];
	Vector4f* estim = (Vector4f*)d_params;
	LaunchKernel(numspots, [=]__device__(int i) {
		const float* smp = &d_sample[i*roisize*roisize];

		Vector3f e = ComputePhasorEstim(smp, roisize,roisize );
		estim[i] = { e[0],e[1],e[2],0.0f };

		d_iterations[i] = 0;
	}, 0, stream, 1024);
}


CDLL_EXPORT Estimator * CreateCenterOfMassEstimator(int roisize, Context* ctx)
{
	auto* p = new cuEstimatorWrapper(new CenterOfMassEstimator(roisize));
	if (ctx) p->SetContext(ctx);
	return p;
}

CDLL_EXPORT Estimator* CreatePhasorEstimator(int roisize, Context* ctx)
{
	auto* p = new cuEstimatorWrapper(new PhasorEstimator(roisize));
	if (ctx) p->SetContext(ctx);
	return p;
}

CDLL_EXPORT void EstimateIntensityAndBackground(const float* imageData, const float* psf, Vector2f* IBg, Vector2f* IBg_crlb,
	int numspots, int numsamples, int maxIterations, bool cuda)
{
	//std::vector<float> psf_space(imgw * imgw * numspots);

	LevMarSettings<2> lms{ {1.0f, 0.0f}, { 1e9, 1e9}, { 1.0f, 1.0f}, maxIterations, true };

	palala_for(numspots, cuda, 
		PLL_FN(int i, const float* psf, const float* imageData, Vector2f * IBg, Vector2f * IBg_crlb) 
	{
		const float* spot_img = &imageData[numsamples * i];
		const float* spot_psf = &psf[numsamples * i];
		IntensityBgModel model({ numsamples, 1 }, spot_psf);

		auto r = LevMarOptimize(spot_img,{ 1,0 }, model, lms);
		IBg[i] = r.estimate;
		IBg_crlb[i] = ComputeCRLB(ComputeFisherMatrix(model, r.estimate));
	}, const_array(psf, numsamples * numspots),
		const_array(imageData, numspots * numsamples),
		out_array(IBg, numspots),
		out_array(IBg_crlb, numspots));
}


