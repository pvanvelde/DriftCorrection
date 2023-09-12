// Estimator C API
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "Estimator.h"
#include "StringUtils.h"
#include "CudaUtils.h"
#include "SolveMatrix.h"

EstimatorBase::EstimatorBase(
	const std::vector<int>& sampleSize, 
	int numConst, 
	int diagsize, 
	const char * paramFormat, 
	param_limits_t limits, Context* ctx
): 
	paramFormat(paramFormat), 
	sampleSize(sampleSize),
	sampleCount(1), 
	numConstants(numConst), 
	numParams(limits[0].size()), 
	limits(limits), 
	diagsize(diagsize), 
	lm_params{},
	lm_step_coeff(limits[0].size(), 1.0f),
	ContextObject(ctx)
{
	for (int s : sampleSize) sampleCount *= s;
	paramNames = StringSplit(this->paramFormat, ',');
}

EstimatorBase::~EstimatorBase()
{
}

int EstimatorBase::ParamIndex(const char * name)
{
	for (int i = 0; i < paramNames.size(); i++)
		if (paramNames[i] == name)
			return i;
	return -1;
}



CDLL_EXPORT void Estimator_Delete(Estimator * e)
{
	delete e;
}

CDLL_EXPORT const char * Estimator_ParamFormat(Estimator * e)
{
	return e->ParamFormat();
}

CDLL_EXPORT void Estimator_GetProperties(Estimator* psf, EstimatorProperties& props)
{
	props.numConst = psf->NumConstants();
	props.numDiag = psf->DiagSize();
	props.numParams = psf->NumParams();
	props.sampleCount = psf->SampleCount();
	props.sampleIndexDims = psf->SampleIndexDims();
}

CDLL_EXPORT void Estimator_SampleDims(Estimator* e, int* dims)
{
	for (int i = 0; i < e->SampleIndexDims(); i++)
		dims[i] = e->SampleSize()[i];
}

CDLL_EXPORT void Estimator_GetParamLimits(Estimator* estim, float* min, float* max)
{
	try {
		auto lim = estim->GetLimits();
		for (int i = 0; i < estim->NumParams(); i++) {
			min[i] = lim[0][i];
			max[i] = lim[1][i];
		}
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}
}



CDLL_EXPORT void Estimator_SetParamLimits(Estimator* estim, float* min, float* max)
{
	try {
		param_limits_t lim
		{ 
			std::vector<float>(estim->NumParams()),
			std::vector<float>(estim->NumParams()) 
		};

		for (int i = 0; i < estim->NumParams(); i++) {
			lim[0][i] = min[i];
			lim[1][i] = max[i];
		}
		estim->SetLimits(lim);
	}
	catch (const std::runtime_error& e) {
		DebugPrintf("%s\n", e.what());
	}
}


CDLL_EXPORT void Estimator_GetLMParams(Estimator* estim, LMParams& lmParams, float* stepcoeff)
{
	try {
		auto r = estim->GetLMParams();
		lmParams = r.first;
		for (int i = 0; i < estim->NumParams(); i++)
			stepcoeff[i] = r.second[i];
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}
}

CDLL_EXPORT void Estimator_SetLMParams(Estimator* estim, const LMParams& lmParams, const float* stepcoeff)
{
	try {
		estim->SetLMParams(lmParams, std::vector<float>(stepcoeff, stepcoeff+estim->NumParams()));
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}
}


CDLL_EXPORT void Estimator_ComputeExpectedValue(Estimator * e, int numspots, const float* params, const float* constants, const int* spot_pos, float * ev)
{
	//DebugPrintf("Estimator_ExpVal: %d spots. fmt: %s\n", numspots, e->ParamFormat());
	if (numspots == 0)
		return;
	try {
		e->ExpectedValue(ev, params, constants, spot_pos, numspots);
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}
}

CDLL_EXPORT void Estimator_Estimate(Estimator * e, int numspots, const float * sample, const float* constants, 
	const int* spot_pos, const float * initial, float * params,  float *diagnostics, int* iterations, 
	float* trace, int traceBufLen)
{
	//DebugPrintf("Estimator_Estimate: %d spots. fmt: %s\n", numspots, e->ParamFormat());
	if (numspots == 0)
		return;
	try {
		e->Estimate(sample, constants, spot_pos, initial, params, diagnostics, iterations, numspots, trace, traceBufLen);
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}
}


CDLL_EXPORT void Estimator_ComputeDerivatives(Estimator * e, int numspots, const float * params, const float* constants,
	const int* spot_pos, float * psf_deriv, float * ev)
{
	if (numspots == 0)
		return;
	//DebugPrintf("Estimator_ComputeDerivatives: numspots=%d. fmt=%s\n", numspots, e->ParamFormat());
	try {
		e->Derivatives(psf_deriv, ev, params, constants, spot_pos, numspots);
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}

}

CDLL_EXPORT void Estimator_ChiSquareAndCRLB(Estimator* estim, const float* params, const float* sample, const float* h_const, const int* spot_pos, int numspots, float* crlb, float* chisq)
{
	if (numspots == 0)
		return;

	try {
		estim->ChiSquareAndCRLB(params, sample, h_const, spot_pos, crlb, chisq, numspots);
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}
}
