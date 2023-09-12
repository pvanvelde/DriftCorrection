#include <array>
#include <tuple>
#include <cassert>

#include "SolveMatrix.h"
#include "MathUtils.h"
#include "ThreadUtils.h"

#include "Estimators/CSpline/CubicSplinePSF.h"
#include "DebugImageCallback.h"
//#include "TIFFReadWrite.h"

#include "StringUtils.h"
#include "CudaUtils.h"
#include <stdio.h>

#include "RandomDistributions.h"
#include "Estimators/Estimator.h"
#include "Estimators/Gaussian/GaussianPSF.h"

#include "Estimators/SIMFLUX/SIMFLUX_PSF.h"

#include "GetPreciseTime.h"

#include "ROIQueue.h"
#include "palala.h"


const float pi = 3.141592653589793f;

void cudaInit();

void testSimfluxFitter()
{
	const int N = 1;
	const int roisize = 10;
	const int npat = 6;
	float sigma = 1.8f;
	auto ctx = Context_Create();
	auto sf_psf = SIMFLUX_Gauss2D_CreateEstimator(npat, sigma, sigma, roisize, npat, ctx);

	auto psf = Gauss2D_CreatePSF_XYIBg(roisize, sigma, sigma, false, ctx);

	float expval[roisize * roisize * npat];
	float r = 1.0f / npat;
	float d = 0.9;
	float k = 1.8f;

	float mod[6 * npat] = { // kx,ky,kz,depth,phase,relint
		k, 0, 0, d, 0.0f, r,
		0, k, 0, d, 0.0f, r,
		k, 0, 0, d, 2 * pi / 3, r,
		0, k, 0, d, 2 * pi / 3, r,
		k, 0, 0, d, 4 * pi / 3, r,
		0, k, 0, d, 4 * pi / 3, r,
	};

	float params[] = { roisize * 0.5f-0.5f, roisize * 0.5f-0.5f, 400.0f, 1.0f };
	int spot_pos[] = { 0,0,0 };

	//psf->ExpectedValue(expval, params, mod, spot_pos, 1);
	sf_psf->ExpectedValue(expval, params, mod, spot_pos, 1);

	float samples[roisize * roisize * npat];
	float s = 0.0f;
	for (int i = 0; i < roisize * roisize * npat; i++) {
		samples[i] = rand_poisson<float>(expval[i]);
		s += samples[i];
	}
	DebugPrintf("s=%f\n", s);

	std::vector<float> summed(roisize * roisize, 0.0f);
	for (int i = 0; i < roisize * roisize; i++) {
		float s = 0.0f;
		for (int j = 0; j < npat; j++)
			s += samples[j * roisize * roisize + i];
		summed[i] = s;
	}


	// plot the first exposure frame
	for (int y = 0; y < roisize; y++) {
		for (int x = 0; x < roisize; x++)
			DebugPrintf("%d ", (int) summed[y * roisize + x]);
		DebugPrintf("\n");
	}

	float initial[] = { roisize * 0.5f, roisize * 0.5f, 1000.0f, 1.0f };
	int tracebuflen = 101;
	std::vector<float> traces(tracebuflen * psf->NumParams());

	int iterations[1];
	std::vector<float> estimated(psf->NumParams() * N);

	LMParams p;
	p.iterations = 100;
	p.normalizeWeights = true;
	sf_psf->SetLMParams(p, { 1.0f,1.0f, 1.0f, 1.0f });

	//psf->Estimate(expval, 0, spot_pos, initial, estimated.data(), 0, iterations, 1, 0, 0);
	//psf->Estimate(summed.data(), 0, spot_pos, initial, estimated.data(), 0, iterations, 1, 0, 0);
	sf_psf->Estimate(samples, mod, spot_pos, initial, estimated.data(), 0, iterations, 1, traces.data(), tracebuflen);

	DebugPrintf("%d iterations", iterations[0]);
	DebugPrintf("x=%f, y=%f, I=%f, bg=%f", estimated[0],estimated[1],estimated[2],estimated[3]);

	Context_Destroy(ctx);
}

int main() {

	cudaInit();

	testSimfluxFitter();

	return;

	int roisize = 16;
	float sigma = 1.8f;
	int n = 20000;
	Estimator* estim = Gauss2D_CreatePSF_XYIBg(roisize, sigma, sigma, true);

#ifdef _DEBUG
	n = 10;
#endif

	std::vector<float> samples(roisize * roisize * n);
	std::vector<Vector4f> params(n);

	for (int i = 0; i < n; i++)
		params[i] = { roisize * 0.5f-1,roisize * 0.5f, 1000, 10 };
	
	estim->ExpectedValue(samples.data(),  (const float*)params.data(), 0, 0, n);

	DebugPrintf("Sampling...");  
	Profile([&]() {
		for (int i = 0; i < n * roisize * roisize; i++)
			samples[i] = rand_poisson<float>(samples[i]);
		});
	DeviceArray<float> d_samples(samples);
	DeviceArray<Vector4f> d_params(params);

	std::vector<Vector4f> initial(n);
	for (int i = 0; i < n; i++)
		initial[i] = { roisize * 0.5f, roisize * 0.5f, 1000, 0.0f };
	DeviceArray<Vector4f> d_initial(initial);

	DeviceImage<float> d_sampleimg_tr(n, roisize * roisize);
	DeviceImage<float> d_sampleimg(roisize * roisize, n);
	d_sampleimg.CopyFromHost(samples.data());

	int numIterations = 20, reps = 100;

	DebugPrintf("Estimating using cuEstimator...");
	double t_ = Profile([&]() {
		cuEstimator* e = estim->Unwrap();
		for (int i = 0; i < reps; i++)
			e->Estimate(d_samples.data(), 0, 0, (const float*)d_initial.data(), (float*)d_params.data(), 0, 0, n, 0, 0, 0);

		cudaStreamSynchronize(0);
	});
	DebugPrintf("cuEstimator: %d spots/s.\n", (int)(n * reps / t_));
	auto h_r = d_params.ToVector();
	for (int i = 0; i < std::min(20, (int)h_r.size()); i++) {
		auto r = h_r[i];
		DebugPrintf("cuEstimator results: [%d] x:%.2f, y: %.2f, I: %.0f, b:%.2f\n", i, r[0], r[1], r[2], r[3]);
	}

}

