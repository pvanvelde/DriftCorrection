// SIMFLUX 2D Gaussian PSF Model
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once
#include "CudaUtils.h"
#include "ExcitationModel.h"

#pragma warning(disable : 4503) // decorated name length exceeded, name was truncated

PLL_DEVHOST inline void ComputeImageSums(const float* frames, float* dst, int w, int h, int numframes)
{
	for (int i = 0; i < w*h; i++)
		dst[i] = 0;
	for (int f = 0; f < numframes; f++)
	{
		for (int i = 0; i < w*h; i++)
			dst[i] += frames[f*w*h + i];
	}
}

struct SIMFLUX_Gauss2D_Model
{

	struct Calibration
	{
		SineWaveExcitation epModel;
		float2 sigma;
	};

	typedef float T;
	typedef Vector4f Params;
	enum { K = Params::K };

	typedef Int3 TSampleIndex;

	PLL_DEVHOST SIMFLUX_Gauss2D_Model(int roisize, const Calibration& calib, int startframe, int endframe, int numframes, Int3 roipos)
		: calib(calib), roisize(roisize), startframe(startframe), endframe(endframe), numframes(numframes), roipos(roipos)
	{}

	int roisize;
	int numframes;
	//int numframes; // ep = (frameNum + startPattern) % params.numepp
	int startframe, endframe;
	Calibration calib;
	Int3 roipos;

	PLL_DEVHOST int SampleCount() const { return roisize * roisize * numframes; }

	static PLL_DEVHOST T StopLimit(int k)
	{
		const float deltaStopLimit[] = { 1e-5f, 1e-5f, 1e-2f,1e-4f };
		return deltaStopLimit[k];
	}

	PLL_DEVHOST void CheckLimits(Params& t) const {
		t.elem[0] = clamp(t.elem[0], 2.0f, roisize - 3.0f);
		t.elem[1] = clamp(t.elem[1], 2.0f, roisize - 3.0f);
		t.elem[2] = fmaxf(t.elem[2], 25.0f);
		t.elem[3] = fmaxf(t.elem[3], 0.0f);
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, Vector4f theta) const
	{
		int w = roisize;
		T _1oSq2PiSigmaX = 1.0f / (sqrtf(2 * MATH_PI) * calib.sigma.x);
		T _1oSq2SigmaX = 1.0f / (sqrtf(2) * calib.sigma.x);
		T _1oSq2PiSigmaY = 1.0f / (sqrtf(2 * MATH_PI) * calib.sigma.y);
		T _1oSq2SigmaY = 1.0f / (sqrtf(2) * calib.sigma.y);

		T thetaX = theta[0], thetaY = theta[1], thetaI = theta[2], thetaBg = theta[3];

		int e = roipos[0];
		for (int f = startframe; f < endframe; f++) {
			// compute Q, dQ/dx, dQ/dy
			float Q, dQdx, dQdy;
			calib.epModel.ExcitationPattern(Q, dQdx, dQdy, e, { thetaX+roipos[2],thetaY+roipos[1] });

			for (int y = 0; y < w; y++) {
				// compute Ey,dEy/dy
				T Yexp0 = (y - thetaY + .5f) * _1oSq2SigmaY;
				T Yexp1 = (y - thetaY - .5f) * _1oSq2SigmaY;
				T Ey = 0.5f * erf(Yexp0) - 0.5f * erf(Yexp1);
				T dEy = _1oSq2PiSigmaY * (exp(-Yexp1 * Yexp1) - exp(-Yexp0 * Yexp0));
				for (int x = 0; x < w; x++) {
					// compute Ex,dEx/dx
					T Xexp0 = (x - thetaX + .5f) * _1oSq2SigmaX;
					T Xexp1 = (x - thetaX - .5f) * _1oSq2SigmaX;
					T Ex = 0.5f * erf(Xexp0) - 0.5f * erf(Xexp1);
					T dEx = _1oSq2PiSigmaX * (exp(-Xexp1 * Xexp1) - exp(-Xexp0 * Xexp0));

					const float exc_bg = 1.0f;// calib.epModel.BackgroundPattern(e, x, y);

					// combine
					const float Exy = Ex * Ey;
					const float mu = thetaI * (Q*Exy) + thetaBg * exc_bg;

					const float dmu_x = thetaI * (dQdx * Exy + Q * dEx * Ey);
					const float dmu_y = thetaI * (dQdy * Exy + Q * Ex * dEy);
					const float dmu_I = Q * Exy;
					const float dmu_bg = exc_bg;

					const float jacobian[] = { dmu_x, dmu_y,dmu_I,dmu_bg };

					cb(f*roisize*roisize+y*roisize+x, mu, jacobian);
				}
			}

			e++;
			if (e == calib.epModel.NumPatterns()) e = 0;
		}
	}
};



struct SIMFLUX_Gauss2DAstig_Model
{

	struct Calibration
	{
		SineWaveExcitation epModel;
		Gauss3D_Calibration psf;
	};

	typedef float T;
	typedef Vector5f Params;
	enum { K = Params::K };

	typedef Int3 TSampleIndex;

	PLL_DEVHOST SIMFLUX_Gauss2DAstig_Model(int roisize, const Calibration& calib, int startframe, int endframe, int numframes, Int3 roipos)
		: calib(calib), roisize(roisize), startframe(startframe), endframe(endframe), numframes(numframes), roipos(roipos)
	{}

	int roisize;
	int numframes;
	//int numframes; // ep = (frameNum + startPattern) % params.numepp
	int startframe, endframe;
	Calibration calib;
	Int3 roipos;

	PLL_DEVHOST int SampleCount() const { return roisize * roisize * numframes; }

	static PLL_DEVHOST T StopLimit(int k)
	{
		const float deltaStopLimit[] = { 1e-4f, 1e-4f,1e-5f, 1e-5f,1e-6f };
		return deltaStopLimit[k];
	}

	PLL_DEVHOST void CheckLimits(Params& t) const
	{
		const float Border = 2.0f;
		t.elem[0] = clamp(t.elem[0], Border, roisize - Border - 1); // x
		t.elem[1] = clamp(t.elem[1], Border, roisize - Border - 1); // y
		t.elem[2] = clamp(t.elem[2], calib.psf.minz, calib.psf.maxz); // z
		t.elem[3] = fmaxf(t.elem[3], 10.f); //I
		t.elem[4] = fmaxf(t.elem[4], 0.1f); // bg
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, Params theta) const
	{
		int w = roisize;

		T tx = theta[0];
		T ty = theta[1];
		T tz = theta[2];
		T tI = theta[3];
		T tbg = theta[4];

		T s0_x = calib.psf.x[0];
		T gamma_x = calib.psf.x[1];
		T d_x = calib.psf.x[2];
		T A_x = calib.psf.x[3];

		T s0_y = calib.psf.y[0];
		T gamma_y = calib.psf.y[1];
		T d_y = calib.psf.y[2];
		T A_y = calib.psf.y[3];

		T d_y2 = d_y * d_y;
		T d_x2 = d_x * d_x;

		T d_y3 = d_y * d_y * d_y;
		T d_x3 = d_x * d_x * d_x;

		const T tzx = tz - gamma_x; const T tzx2 = tzx * tzx; const T tzx3 = tzx2 * tzx;
		const T tzy = tz - gamma_y; const T tzy2 = tzy * tzy; const T tzy3 = tzy2 * tzy;
		const T tz2 = tz * tz;
		const T sigma_x = s0_x * sqrt(1.0f + tzx2 / d_x2 + A_x * tzx3 / d_x3);
		const T sigma_y = s0_y * sqrt(1.0f + tzy2 / d_y2 + A_y * tzy3 / d_y3);

		const T OneOverSqrt2PiSigma_x = 1.0f / (sqrtf(2 * MATH_PI) * sigma_x);
		const T OneOverSqrt2Sigma_x = 1.0f / (sqrtf(2) * sigma_x);
		const T OneOverSqrt2PiSigma_y = 1.0f / (sqrtf(2 * MATH_PI) * sigma_y);
		const T OneOverSqrt2Sigma_y = 1.0f / (sqrtf(2) * sigma_y);

		float pos[3] = { theta[0] + roipos[2],theta[1] + roipos[1],theta[2] }; // roipos are array indices so their ordering is ZYX

		for (int y = 0; y < w; y++) {
			T Yexp0 = (y - ty + .5f) * OneOverSqrt2Sigma_y;
			T Yexp1 = (y - ty - .5f) * OneOverSqrt2Sigma_y;
			T DeltaY = 0.5f * erf(Yexp0) - 0.5f * erf(Yexp1);
			T dEy = OneOverSqrt2PiSigma_y * (exp(-Yexp1 * Yexp1) - exp(-Yexp0 * Yexp0));
			T G21y = 1 / (sqrt(2.0f * MATH_PI) * sigma_y * sigma_y) * (
				(y - ty - 0.5f) * exp(-(y - ty - 0.5f) * (y - ty - 0.5f) / (2.0f * sigma_y * sigma_y)) -
				(y - ty + 0.5f) * exp(-(y - ty + 0.5f) * (y - ty + 0.5f) / (2.0f * sigma_y * sigma_y)));

			for (int x = 0; x < w; x++) {

				T Xexp0 = (x - tx + .5f) * OneOverSqrt2Sigma_x;
				T Xexp1 = (x - tx - .5f) * OneOverSqrt2Sigma_x;
				T DeltaX = 0.5f * erf(Xexp0) - 0.5f * erf(Xexp1);
				T dEx = OneOverSqrt2PiSigma_x * (exp(-Xexp1 * Xexp1) - exp(-Xexp0 * Xexp0));

				T G21x = 1 / (sqrt(2.0f * MATH_PI) * sigma_x * sigma_x) * (
					(x - tx - 0.5f) * exp(-(x - tx - 0.5f) * (x - tx - 0.5f) / (2.0f * sigma_x * sigma_x)) -
					(x - tx + 0.5f) * exp(-(x - tx + 0.5f) * (x - tx + 0.5f) / (2.0f * sigma_x * sigma_x)));

				T dMuSigmaX = DeltaY * G21x;
				T dMuSigmaY = DeltaX * G21y;

				T dSigmaXThetaZ = s0_x * (2 * tzx / d_x2 + A_x * 3 * tzx2 / d_x3) /
					(2 * sqrt(1 + tzx2 / d_x2 + A_x * tzx3 / d_x3));
				T dSigmaYThetaZ = s0_y * (2 * tzy / d_y2 + A_y * 3 * tzy2 / d_y3) /
					(2 * sqrt(1 + tzy2 / d_y2 + A_y * tzy3 / d_y3));

				T psf_dz = dMuSigmaX * dSigmaXThetaZ + dMuSigmaY * dSigmaYThetaZ;

				int e = roipos[0];
				for (int f = startframe; f < endframe; f++) {
					// compute Q, dQ/dx, dQ/dy
					float exc, excDeriv[3]; 
					calib.epModel.ExcitationPattern(exc, excDeriv, e, pos);

					T dmu_dx = tI * (excDeriv[0] * DeltaX * DeltaY +exc * dEx * DeltaY);
					T dmu_dy = tI * (excDeriv[1] * DeltaX * DeltaY +  exc * DeltaX * dEy);
					T dmu_dz = tI * (excDeriv[2] * DeltaX * DeltaY +  exc * psf_dz);

					T mu = tbg + tI * exc * DeltaX * DeltaY;
					T dmu_dI0 = exc * DeltaX * DeltaY;
					T dmu_dIbg = 1;
					const T jacobian[] = { dmu_dx, dmu_dy, dmu_dz, dmu_dI0, dmu_dIbg};
					cb(f * roisize * roisize + y * roisize + x, mu, jacobian);

					e++;
					if (e == calib.epModel.NumPatterns()) e = 0;
				}
			}

		}
	}
};

//struct SIMFLUX_Gauss2DAstig_Modeltest
//{
//
//	struct Calibration
//	{
//		SineWaveExcitation epModel;
//		Gauss3D_Calibration psf;
//	};
//
//	typedef float T;
//	typedef Vector5f Params;
//	enum { K = Params::K };
//
//	typedef Int3 TSampleIndex;
//
//	PLL_DEVHOST SIMFLUX_Gauss2DAstig_Modeltest(int roisize, const Calibration& calib, int startframe, int endframe, int numframes, Int3 roipos)
//		: calib(calib), roisize(roisize), startframe(startframe), endframe(endframe), numframes(numframes), roipos(roipos)
//	{}
//
//	int roisize;
//	int numframes;
//	//int numframes; // ep = (frameNum + startPattern) % params.numepp
//	int startframe, endframe;
//	Calibration calib;
//	Int3 roipos;
//
//	PLL_DEVHOST int SampleCount() const { return roisize * roisize * numframes; }
//
//	static PLL_DEVHOST T StopLimit(int k)
//	{
//		const float deltaStopLimit[] = { 1e-3f, 1e-3f,1e-4f, 1e-5f,1e-6f };
//		return deltaStopLimit[k];
//	}
//
//	PLL_DEVHOST void CheckLimits(Params& t) const
//	{
//		const float Border = 2.0f;
//		t.elem[0] = clamp(t.elem[0], Border, roisize - Border - 1); // x
//		t.elem[1] = clamp(t.elem[1], Border, roisize - Border - 1); // y
//		t.elem[2] = clamp(t.elem[2], calib.psf.minz, calib.psf.maxz); // z
//		t.elem[3] = fmaxf(t.elem[3], 10.f); //I
//		t.elem[4] = fmaxf(t.elem[4], 0.0f); // bg
//	}
//
//	template<typename TCallback>
//	PLL_DEVHOST void ComputeDerivatives(TCallback cb, Params theta) const
//	{
//		int w = roisize;
//
//		T tx = theta[0];
//		T ty = theta[1];
//		T tz = theta[2];
//		T tI = theta[3];
//		T tbg = theta[4];
//
//		T s0_x = calib.psf.x[0];
//		T gamma_x = calib.psf.x[1];
//		T d_x = calib.psf.x[2];
//		T A_x = calib.psf.x[3];
//
//		T s0_y = calib.psf.y[0];
//		T gamma_y = calib.psf.y[1];
//		T d_y = calib.psf.y[2];
//		T A_y = calib.psf.y[3];
//
//		T d_y2 = d_y * d_y;
//		T d_x2 = d_x * d_x;
//
//		T d_y3 = d_y * d_y * d_y;
//		T d_x3 = d_x * d_x * d_x;
//
//		const T tzx = tz - gamma_x; const T tzx2 = tzx * tzx; const T tzx3 = tzx2 * tzx;
//		const T tzy = tz - gamma_y; const T tzy2 = tzy * tzy; const T tzy3 = tzy2 * tzy;
//
//		const T sigma_x = s0_x * sqrt(1.0f + tzx2 / d_x2 + A_x * tzx3 / d_x3);
//		const T sigma_y = s0_y * sqrt(1.0f + tzy2 / d_y2 + A_y * tzy3 / d_y3);
//
//		const T OneOverSqrt2PiSigma_x = 1.0f / (sqrtf(2 * MATH_PI) * sigma_x);
//		const T OneOverSqrt2Sigma_x = 1.0f / (sqrtf(2) * sigma_x);
//		const T OneOverSqrt2PiSigma_y = 1.0f / (sqrtf(2 * MATH_PI) * sigma_y);
//		const T OneOverSqrt2Sigma_y = 1.0f / (sqrtf(2) * sigma_y);
//
//		float pos[3] = { theta[0] + roipos[2],theta[1] + roipos[1],theta[2] }; // roipos are array indices so their ordering is ZYX
//
//		for (int y = 0; y < w; y++) {
//			T Yexp0 = (y - ty + .5f) * OneOverSqrt2Sigma_y;
//			T Yexp1 = (y - ty - .5f) * OneOverSqrt2Sigma_y;
//			T DeltaY = 0.5f * erf(Yexp0) - 0.5f * erf(Yexp1);
//			T dEy = OneOverSqrt2PiSigma_y * (exp(-Yexp1 * Yexp1) - exp(-Yexp0 * Yexp0));
//			T G21y = 1 / (sqrt(2.0f * MATH_PI) * sigma_y * sigma_y) * (
//				(y - ty - 0.5f) * exp(-(y - ty - 0.5f) * (y - ty - 0.5f) / (2.0f * sigma_y * sigma_y)) -
//				(y - ty + 0.5f) * exp(-(y - ty + 0.5f) * (y - ty + 0.5f) / (2.0f * sigma_y * sigma_y)));
//
//			for (int x = 0; x < w; x++) {
//
//				T Xexp0 = (x - tx + .5f) * OneOverSqrt2Sigma_x;
//				T Xexp1 = (x - tx - .5f) * OneOverSqrt2Sigma_x;
//				T DeltaX = 0.5f * erf(Xexp0) - 0.5f * erf(Xexp1);
//				T dEx = OneOverSqrt2PiSigma_x * (exp(-Xexp1 * Xexp1) - exp(-Xexp0 * Xexp0));
//
//				T G21x = 1 / (sqrt(2.0f * MATH_PI) * sigma_x * sigma_x) * (
//					(x - tx - 0.5f) * exp(-(x - tx - 0.5f) * (x - tx - 0.5f) / (2.0f * sigma_x * sigma_x)) -
//					(x - tx + 0.5f) * exp(-(x - tx + 0.5f) * (x - tx + 0.5f) / (2.0f * sigma_x * sigma_x)));
//
//				T dMuSigmaX = DeltaY * G21x;
//				T dMuSigmaY = DeltaX * G21y;
//
//				T dSigmaXThetaZ = s0_x * (2 * tzx / d_x2 + A_x * 3 * tzx2 / d_x3) /
//					(2 * sqrt(1 + tzx2 / d_x2 + A_x * tzx3 / d_x3));
//				T dSigmaYThetaZ = s0_y * (2 * tzy / d_y2 + A_y * 3 * tzy2 / d_y3) /
//					(2 * sqrt(1 + tzy2 / d_y2 + A_y * tzy3 / d_y3));
//
//				T psf_dz = dMuSigmaX * dSigmaXThetaZ + dMuSigmaY * dSigmaYThetaZ;
//
//				int e = roipos[0];
//				for (int f = startframe; f < endframe; f++) {
//					// compute Q, dQ/dx, dQ/dy
//					float exc, excDeriv[3];
//					calib.epModel.ExcitationPattern(exc, excDeriv, e, pos);
//
//					T dmu_dx = tI * (excDeriv[0] + exc); // *DeltaX* DeltaY + exc * dEx * DeltaY);
//					T dmu_dy = tI * (excDeriv[1] + exc); // DeltaX * DeltaY + exc * DeltaX * dEy);
//					T dmu_dz = tI * (excDeriv[2] + exc); // * DeltaY + exc * psf_dz);
//
//					T mu = tbg + tI * exc; // * DeltaX * DeltaY;
//					T dmu_dI0 = exc; //* DeltaX * DeltaY;
//					T dmu_dIbg = 1;
//					const T jacobian[] = { dmu_dx, dmu_dy, dmu_dz, dmu_dI0, dmu_dIbg };
//					cb(f * roisize * roisize + y * roisize + x, mu, jacobian);
//
//					e++;
//					if (e == calib.epModel.NumPatterns()) e = 0;
//				}
//			}
//
//		}
//	}
//};
//
