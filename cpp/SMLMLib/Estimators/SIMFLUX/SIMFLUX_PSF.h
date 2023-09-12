// SIMFLUX Estimator model
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "DLLMacros.h"
#include "CameraCalibration.h"
#include "simflux/SIMFLUX.h"

class Estimator;
struct SIMFLUX_Modulation;

CDLL_EXPORT Estimator* SIMFLUX_Gauss2DAstig_CreateEstimator(int num_patterns, const Gauss3D_Calibration& calib,
	int roisize, int numframes, Context* ctx);

CDLL_EXPORT Estimator* SIMFLUX_Gauss2D_CreateEstimator(int num_patterns, float sigmaX, float sigmaY,
	int roisize, int numframes, Context* ctx);


