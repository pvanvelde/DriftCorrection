# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 23:20:06 2022

@author: Pieter
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from typing import List, Tuple, Union, Optional
from torch import Tensor


@torch.jit.script
def gauss_psf_2D(theta, numpixels: int):
    """
    theta: [x,y,N,bg,sigma].T
    """
    pi = 3.141592653589793  # torch needs to include pi

    x, y, N, bg, sigma = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3], theta[:, 4]
    pixelpos = torch.arange(0, numpixels, device=theta.device)

    OneOverSqrt2PiSigma = (1.0 / (torch.sqrt(2 * pi) * sigma))[:, None, None]
    OneOverSqrt2Sigma = (1.0 / (torch.sqrt(2) * sigma))[:, None, None]

    # Pixel centers
    Xc = pixelpos[None, None, :]
    Yc = pixelpos[None, :, None]
    Xexp0 = (Xc - x[:, None, None] + 0.5) * OneOverSqrt2Sigma
    Xexp1 = (Xc - x[:, None, None] - 0.5) * OneOverSqrt2Sigma
    Ex = 0.5 * torch.erf(Xexp0) - 0.5 * torch.erf(Xexp1)
    dEx = OneOverSqrt2PiSigma * (torch.exp(-Xexp1 ** 2) - torch.exp(-Xexp0 ** 2))
    dEx_dSigma = (torch.exp(-Xexp1 ** 2) * Xexp1 - torch.exp(-Xexp0 ** 2) * Xexp0) / torch.sqrt(pi)

    Yexp0 = (Yc - y[:, None, None] + 0.5) * OneOverSqrt2Sigma
    Yexp1 = (Yc - y[:, None, None] - 0.5) * OneOverSqrt2Sigma
    Ey = 0.5 * torch.erf(Yexp0) - 0.5 * torch.erf(Yexp1)
    dEy = OneOverSqrt2PiSigma * (torch.exp(-Yexp1 ** 2) - torch.exp(-Yexp0 ** 2))
    dEy_dSigma = (torch.exp(-Yexp1 ** 2) * Yexp1 - torch.exp(-Yexp0 ** 2) * Yexp0) / torch.sqrt(pi)

    mu = N[:, None, None] * Ex * Ey + bg[:, None, None]
    dmu_x = N[:, None, None] * Ey * dEx
    dmu_y = N[:, None, None] * Ex * dEy
    dmu_I = Ex * Ey
    dmu_bg = 1 + mu * 0
    dmu_sigma = N[:, None, None] * (Ex * dEy_dSigma + dEx_dSigma * Ey)

    deriv = torch.stack((dmu_x, dmu_y, dmu_I, dmu_bg, dmu_sigma), -1)
    return mu, deriv


@torch.jit.script
def gauss_psf_2D_fixed_sigma(theta, roisize: int, sigma: float):
    sigma_ = torch.ones((len(theta), 1), device=theta.device) * sigma
    theta_ = torch.cat((theta, sigma_), -1)

    mu, jac = gauss_psf_2D(theta_, roisize)
    return mu, jac[..., :-1]


class Gaussian2DFixedSigmaPSF(torch.nn.Module):
    def __init__(self, roisize, sigma):
        super().__init__()
        self.roisize = roisize
        self.sigma = sigma

    def forward(self, x, const_: Optional[Tensor] = None):
        return gauss_psf_2D_fixed_sigma(x, self.roisize, self.sigma)


@torch.jit.script
def gauss_psf_2D_astig(theta: Tensor, numpixels: int, calib: List[List[float]]):
    # joe
    tx = theta[:, 0, None, None]
    ty = theta[:, 1, None, None]
    tz = theta[:, 2, None, None]
    tI = theta[:, 3, None, None]
    tbg = theta[:, 4, None, None]

    sqrt2pi = 2.5066282746310002
    sqrt2 = 1.4142135623730951

    s0_x, gamma_x, d_x, A_x = calib[0]
    s0_y, gamma_y, d_y, A_y = calib[1]

    tzx = tz - gamma_x
    tzy = tz - gamma_y
    sigma_x = s0_x * torch.sqrt(1 + tzx ** 2 / d_x ** 2 + A_x * tzx ** 3 / d_x ** 3)
    sigma_y = s0_y * torch.sqrt(1 + tzy ** 2 / d_y ** 2 + A_y * tzy ** 3 / d_y ** 3)

    OneOverSqrt2PiSigma_x = 1 / (sqrt2pi * sigma_x)
    OneOverSqrt2Sigma_x = 1 / (sqrt2 * sigma_x)
    OneOverSqrt2PiSigma_y = 1 / (sqrt2pi * sigma_y)
    OneOverSqrt2Sigma_y = 1 / (sqrt2 * sigma_y)

    pixelpos = torch.arange(0, numpixels, device=theta.device)
    Xc = pixelpos[None, None, :]
    Yc = pixelpos[None, :, None]

    # Pixel centers
    Xexp0 = (Xc - tx + 0.5) * OneOverSqrt2Sigma_x
    Xexp1 = (Xc - tx - 0.5) * OneOverSqrt2Sigma_x
    Ex = 0.5 * torch.erf(Xexp0) - 0.5 * torch.erf(Xexp1)
    dEx = OneOverSqrt2PiSigma_x * (torch.exp(-Xexp1 ** 2) - torch.exp(-Xexp0 ** 2))

    Yexp0 = (Yc - ty + 0.5) * OneOverSqrt2Sigma_y
    Yexp1 = (Yc - ty - 0.5) * OneOverSqrt2Sigma_y
    Ey = 0.5 * torch.erf(Yexp0) - 0.5 * torch.erf(Yexp1)
    dEy = OneOverSqrt2PiSigma_y * (torch.exp(-Yexp1 ** 2) - torch.exp(-Yexp0 ** 2))

    G21y = 1 / (sqrt2pi * sigma_y * sigma_y) * (
            (Yc - ty - 0.5) * torch.exp(-(Yc - ty - 0.5) * (Yc - ty - 0.5) / (2.0 * sigma_y * sigma_y)) -
            (Yc - ty + 0.5) * torch.exp(-(Yc - ty + 0.5) * (Yc - ty + 0.5) / (2.0 * sigma_y * sigma_y)))

    mu = tbg + tI * Ex * Ey
    dmu_dx = tI * dEx * Ey
    dmu_dy = tI * dEy * Ex

    G21x = 1 / (sqrt2pi * sigma_x * sigma_x) * (
            (Xc - tx - 0.5) * torch.exp(-(Xc - tx - 0.5) * (Xc - tx - 0.5) / (2 * sigma_x * sigma_x)) -
            (Xc - tx + 0.5) * torch.exp(-(Xc - tx + 0.5) * (Xc - tx + 0.5) / (2 * sigma_x * sigma_x)))

    dMuSigmaX = tI * Ey * G21x
    dMuSigmaY = tI * Ex * G21y

    dSigmaXThetaZ = (s0_x * (2 * tzx / d_x ** 2 + A_x * 3 * tzx ** 2 / d_x ** 3) /
                     (2 * torch.sqrt(1 + tzx ** 2 / d_x ** 2 + A_x * tzx ** 3 / d_x ** 3)))
    dSigmaYThetaZ = (s0_y * (2 * tzy / d_y ** 2 + A_y * 3 * tzy ** 2 / d_y ** 3) /
                     (2 * torch.sqrt(1 + tzy ** 2 / d_y ** 2 + A_y * tzy ** 3 / d_y ** 3)))

    dmu_dz = dMuSigmaX * dSigmaXThetaZ + dMuSigmaY * dSigmaYThetaZ

    dmu_dI0 = Ex * Ey
    dmu_dIbg = dmu_dx * 0 + 1

    return mu, torch.stack((dmu_dx, dmu_dy, dmu_dz, dmu_dI0, dmu_dIbg), -1)


class Gaussian2DAstigmaticPSF(torch.nn.Module):
    def __init__(self, roisize, calib):
        super().__init__()
        self.roisize = roisize
        self.calib = calib

    def forward(self, x, const_: Optional[Tensor] = None):
        return gauss_psf_2D_astig(x, self.roisize, self.calib)


# @torch.jit.script
def sf_modulation(xyz: Tensor, mod: Tensor, roipos):
    k = mod[:, :, :3]
    phase = mod[:, :, 4]
    depth = mod[:, :, 3]
    relint = mod[:, :, 5]
    xyz_mod = xyz * 1
    xyz_mod[:, :2] = roipos + xyz[:, 0:2]
    em_phase = ((xyz_mod[:, None] * k).sum(-1)) % (2 * np.pi) - phase

    deriv = depth[:, :, None] * k * torch.cos(em_phase)[:, :, None] * relint[:, :, None]  # [spots,patterns,coords]
    intensity = (1 + depth * torch.sin(em_phase)) * relint

    return intensity, deriv


class SIMFLUXModel_vector(torch.nn.Module):
    """
    params: [batchsize, xyzIb]
    mod: [batchsize, frame_window, 6] . format: kx,ky,kz,depth,phase,relint
    """

    # caution: path[0] is reserved for script path (or '' in REPL)

    def __init__(self, psf, roi_pos, divide_bg=False, roisize=16):
        super().__init__()
        self.psf = psf
        self.divide_bg = divide_bg
        self.roiposmod = roi_pos
        self.roisize = roisize

    def forward(self, params, good_array, mod: Optional[Tensor] = None):
        import sys
        sys.path.insert(1, '/vectorpsf')

        import torch
        from vectorize_torch import poissonrate, get_pupil_matrix, initial_guess, thetalimits, compute_crlb, \
            Model_vectorial_psf, LM_MLE
        from config import sim_params
        if mod is None:
            raise ValueError('expecting modulation patterns')

        # mu = bg + I * P(xyz) * PSF(xyz)
        xyz = params[:, :3]
        I = params[:, [3]]
        bg = params[:, [4]]

        if self.divide_bg:
            bg /= mod.shape[1]

        mod_intensity, mod_deriv = sf_modulation(xyz, mod, self.roiposmod[good_array, :])

        normalized_psf_params = torch.cat((xyz,
                                           torch.ones(I.shape, device=params.device),
                                           torch.zeros(I.shape, device=params.device)), -1)
        # normalized_psf_params = torch.cat((xyz,
        #                                    I,
        #                                    bg), -1)
        # psf_ev, psf_deriv = self.psf(normalized_psf_params)
        # start vectorial stuff
        paramsss = sim_params()
        dev = 'cuda'
        zstack = False
        # region get values from paramsss
        NA = paramsss.NA
        refmed = paramsss.refmed
        refcov = paramsss.refcov
        refimm = paramsss.refimm
        refimmnom = paramsss.refimmnom
        Lambda = paramsss.Lambda
        Npupil = paramsss.Npupil
        abberations = torch.from_numpy(paramsss.abberations).to(dev)
        zvals = torch.from_numpy(paramsss.zvals).to(dev)
        zspread = torch.tensor(paramsss.zspread).to(dev)
        numparams = paramsss.numparams
        numparams_fit = 5
        K = paramsss.K
        Mx = paramsss.Mx
        My = paramsss.My
        Mz = paramsss.Mz

        pixelsize = paramsss.pixelsize
        Ax = torch.tensor(paramsss.Axmt).to(dev)
        Bx = torch.tensor(paramsss.Bxmt).to(dev)
        Dx = torch.tensor(paramsss.Dxmt).to(dev)
        Ay = torch.tensor(paramsss.Aymt).to(dev)
        By = torch.tensor(paramsss.Bymt).to(dev)
        Dy = torch.tensor(paramsss.Dymt).to(dev)

        N = paramsss.cztN
        M = paramsss.cztM
        L = paramsss.cztL

        thetamin, thetamax = thetalimits(abberations, Lambda, Mx, My, pixelsize, zspread, dev, zstack=False)
        param_range = torch.concat((thetamin[..., None], thetamax[..., None]), dim=1)

        normalized_psf_params[:, 0:2] = (normalized_psf_params[:, 0:2] - self.roisize / 2) * pixelsize
        normalized_psf_params[:, 2] = normalized_psf_params[:, 2] * 1000

        model = Model_vectorial_psf()

        wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix = get_pupil_matrix(NA, zvals,
                                                                                              refmed,
                                                                                              refcov,
                                                                                              refimm,
                                                                                              refimmnom,
                                                                                              Lambda,
                                                                                              Npupil,
                                                                                              abberations, dev)

        psf_ev, psf_deriv = model.forward(NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil, abberations, 0,
                                          0, K, N,
                                          M, L, Ax, Bx,
                                          Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, normalized_psf_params, dev,
                                          zstack,
                                          wavevector, wavevectorzimm, all_zernikes, PupilMatrix)
        psf_deriv[:, :, :, :2] = psf_deriv[:, :, :, :2] / pixelsize
        psf_deriv[:, :, :, 2] = psf_deriv[:, :, :, 2] / 1000

        # end vectorial stuff

        mod_intensity = torch.ones(mod_intensity.size())
        mod_deriv = torch.ones(mod_deriv.size())
        phot = I[:, None, None]
        mu = bg[:, None, None] + mod_intensity[:, :, None, None] * phot * psf_ev[:, None, :, :]

        deriv_xyz = phot[..., None] * (mod_deriv[:, :, None, None] * psf_ev[:, None, ..., None] +
                                       mod_intensity[:, :, None, None, None] * psf_deriv[:, None, ..., :3])

        deriv_I = psf_ev[:, None] * mod_intensity[:, :, None, None]

        deriv_bg = torch.ones(deriv_I.shape, device=params.device)

        return mu, torch.cat((deriv_xyz,
                              deriv_I[..., None],
                              deriv_bg[..., None]), -1)


class SIMFLUXModel(torch.nn.Module):
    """
    params: [batchsize, xyzIb]
    mod: [batchsize, frame_window, 6] . format: kx,ky,kz,depth,phase,relint
    """

    def __init__(self, psf, roi_pos, divide_bg=False):
        super().__init__()
        self.psf = psf
        self.divide_bg = divide_bg
        self.roiposmod = roi_pos

    def forward(self, params, roi_pos, mod: Optional[Tensor] = None):
        if mod is None:
            raise ValueError('expecting modulation patterns')

        # mu = bg + I * P(xyz) * PSF(xyz)
        xyz = params[:, :3]
        I = params[:, [3]]
        bg = params[:, [4]]

        if self.divide_bg:
            bg /= mod.shape[1]
        altered_xyz = xyz * 1
        altered_xyz[:, 2] = xyz[:, 2] + 0.0000001

        mod_intensity, mod_deriv = sf_modulation(xyz, mod, self.roiposmod)
        mod_intensityadj, mod_derivadj = sf_modulation(altered_xyz, mod, self.roiposmod)
        normalized_psf_params = torch.cat((xyz,
                                           torch.ones(I.shape, device=params.device),
                                           torch.zeros(I.shape, device=params.device)), -1)
        # normalized_psf_params = torch.cat((xyz,
        #                                    I,
        #                                    bg), -1)
        psf_ev, psf_deriv = self.psf(normalized_psf_params)
        normalized_psf_params_adj = normalized_psf_params * 1
        normalized_psf_params_adj[:, 2] = normalized_psf_params_adj[:, 2] + 0.0000001
        psf_evadj, psfderiv_adj = self.psf(normalized_psf_params_adj)

        phot = I[:, None, None]
        mu = bg[:, None, None] + mod_intensity[:, :, None, None] * phot * psf_ev[:, None, :, :]

        mu_adj = bg[:, None, None] + mod_intensityadj[:, :, None, None] * phot * psf_evadj[:, None, :, :]

        dmu = (mu_adj - mu) / 0.0000001
        dmu = dmu.cpu().detach().numpy()
        deriv_xyz = phot[..., None] * (mod_deriv[:, :, None, None] * psf_ev[:, None, ..., None] +
                                       mod_intensity[:, :, None, None, None] * psf_deriv[:, None, ..., :3])

        deriv_I = psf_ev[:, None] * mod_intensity[:, :, None, None]

        deriv_bg = torch.ones(deriv_I.shape, device=params.device)

        return mu, torch.cat((deriv_xyz,
                              deriv_I[..., None],
                              deriv_bg[..., None]), -1)


# @torch.jit.script
class ModulatedIntensitiesModel:
    """
    params: [batchsize, ...]
    mod: [batchsize, frame_window, 6] . format: kx,ky,kz,depth,phase,relint

    basically simflux_model but with all psf derivatives set to zero, so they don't contribute any FI.

    psf_ev is now just 1.
    This brings up a problem for the background, which is defined per pixel.
    As a hack, bg_factor can be set to the number of pixels that the PSF roughly covers,
    divided by number of patterns (as in normal simflux)
    """

    def __init__(self, roi_pos, bg_factor: float = 1):
        # super().__init__()
        self.bg_factor = bg_factor
        self.roiposmod = roi_pos

    def __call__(self, params, mod: Optional[Tensor] = None):
        return self.forward(params, mod)

    def forward(self, params, roi_pos, mod: Optional[Tensor] = None):
        if mod is None:
            raise ValueError('expecting modulation patterns')
        xyz = params[:, :3]
        I = params[:, [3]]
        bg = params[:, [4]]

        mod_intensity, mod_deriv = sf_modulation(xyz, mod, roi_pos)

        mu = bg + mod_intensity * I

        deriv_xyz = I[..., None] * mod_deriv
        deriv_I = mod_intensity
        deriv_bg = torch.ones(deriv_I.shape, device=params.device)

        return mu, torch.cat((deriv_xyz,
                              deriv_I[..., None],
                              deriv_bg[..., None]), -1)


def compute_crlb(mu: Tensor, jac: Tensor, *, skip_axes: List[int] = []):
    """
    Compute crlb from expected value and per pixel derivatives.
    mu: [N, H, W]
    jac: [N, H,W, coords]
    """
    if not isinstance(mu, torch.Tensor):
        mu = torch.Tensor(mu)

    if not isinstance(jac, torch.Tensor):
        jac = torch.Tensor(jac)

    naxes = jac.shape[-1]
    axes = [i for i in range(naxes) if not i in skip_axes]
    jac = jac[..., axes]

    sample_dims = tuple(np.arange(1, len(mu.shape)))

    fisher = torch.matmul(jac[..., None], jac[..., None, :])  # derivative contribution
    fisher = fisher / mu[..., None, None]  # px value contribution
    fisher = fisher.sum(sample_dims)

    crlb = torch.zeros((len(mu), naxes), device=mu.device)
    crlb[:, axes] = torch.sqrt(torch.diagonal(torch.inverse(fisher), dim1=1, dim2=2))
    return crlb


# @torch.jit.script
def lm_alphabeta(mu, jac, smp):
    """
    mu: [batchsize, numsamples]
    jac: [batchsize, numsamples, numparams]
    smp: [batchsize, numsamples]
    """
    # assert np.array_equal(smp.shape, mu.shape)
    sampledims = [i for i in range(1, len(smp.shape))]

    invmu = 1.0 / torch.clip(mu, min=1e-9)
    af = smp * invmu ** 2

    jacm = torch.matmul(jac[..., None], jac[..., None, :])
    alpha = jacm * af[..., None, None]
    alpha = alpha.sum(sampledims)

    beta = (jac * (smp * invmu - 1)[..., None]).sum(sampledims)
    return alpha, beta


@torch.jit.script
def lm_update(cur, mu, jac, smp, lambda_: float, param_range_min_max: Tensor, scale_old=torch.Tensor(1).to('cuda')):
    """
    Separate some of the calculations to speed up with jit script
    """
    alpha, beta = lm_alphabeta(mu, jac, smp)
    scale_old = scale_old.to(device=cur.device)
    K = cur.shape[-1]

    if True:  # scale invariant. Helps when parameter scales are quite different
        # For a matrix A, (element wise A*A).sum(0) is the same as diag(A^T * A)
        scale = (alpha * alpha).sum(1)
        # scale /= scale.mean(1, keepdim=True) # normalize so lambda scale is not model dependent
        # assert torch.isnan(scale).sum()==0
        if scale_old.size() != torch.Size([1]):
            scale = torch.maximum(scale, scale_old)

        alpha += lambda_ * scale[:, :, None] * torch.eye(K, device=smp.device)[None]
    else:
        # regular LM, non scale invariant
        alpha += lambda_ * torch.eye(K, device=smp.device)[None]

    steps = torch.linalg.solve(alpha, beta)

    assert torch.isnan(cur).sum() == 0
    assert torch.isnan(steps).sum() == 0

    cur = cur + steps
    cur = torch.maximum(cur, param_range_min_max[None, :, 0])
    cur = torch.minimum(cur, param_range_min_max[None, :, 1])
    return cur, scale


def lm_mle(model, initial, smp, param_range_min_max, roi_pos_, mod_, iterations=50, lambda_=1, store_traces=False):
    """
        model:
            function that takes parameter array [batchsize, num_parameters]
            and returns (expected_value, jacobian).
            Expected value has shape [batchsize, sample dims...]
            Jacobian has shape [batchsize, sample dims...,num_parameters]

        initial: [batchsize, num_parameters]

        return value is a tuple with:
            estimates [batchsize,num_parameters]
            traces [iterations, batchsize, num_parameters]
    """
    dev = 'cuda'
    if not isinstance(initial, torch.Tensor):
        initial = torch.Tensor(initial).to(smp.device)
    cur = (initial * 1).type(torch.double)

    if not isinstance(param_range_min_max, torch.Tensor):
        param_range_min_max = torch.Tensor(param_range_min_max).to(smp.device)

    traces = torch.zeros((iterations + 1, cur.size()[0], cur.size()[1]), device=dev).type(torch.double)
    traces[0, :, :] = cur

    assert len(smp) == len(initial)
    scale = torch.zeros(cur.size(), device='cuda').type(torch.double)
    tol_ = torch.tensor((1e-3, 1e-3, 1e-5, 0.1, 0.1)).to(dev)
    mu = torch.zeros(smp.size()).to(dev).type(torch.double)
    jac = torch.zeros((smp.size()[0], smp.size()[1], cur.size()[1])).to(dev).type(torch.double)
    tol = torch.ones((cur.size()[0], cur.size()[1])).to(dev) * tol_[None, ...].repeat([cur.size()[0], 1])
    good_array = torch.ones(cur.size()[0]).to(dev).type(torch.bool)
    delta = torch.ones(cur.size()).to(dev).type(torch.double)
    bool_array = torch.ones(cur.size()).to(dev).type(torch.bool)
    i = 0
    flag_tolerance = 0

    while (i < iterations) and (flag_tolerance == 0):
        mu[good_array, :], jac[good_array, :, :] = model.forward(cur[good_array, :], roi_pos_[good_array, :],
                                                                 mod_[good_array, :])

        cur[good_array, :], scale[good_array, :] = lm_update(cur[good_array, :], mu[good_array, :],
                                                             jac[good_array, :, :], smp[good_array, :], lambda_,
                                                             param_range_min_max, scale[good_array, :])

        traces[i + 1, good_array, :] = cur[good_array, :]
        delta[good_array, :] = torch.absolute(traces[i - 1, good_array, :] - traces[i, good_array, :])

        bool_array[good_array] = (delta[good_array, :] < tol[good_array, :]).type(torch.bool)
        test = torch.sum(bool_array, dim=1)
        good_array = test != 5

        if torch.sum(good_array) == 0:
            flag_tolerance = 1
        i = i + 1
    return cur, traces


class LM_MLE(torch.nn.Module):
    def __init__(self, model, param_range_min_max, iterations: int, lambda_: float):
        """
        model:
            module that takes parameter array [batchsize, num_parameters]
            and returns (expected_value, jacobian).
            Expected value has shape [batchsize, sample dims...]
            Jacobian has shape [batchsize, sample dims...,num_parameters]
        """
        super().__init__()

        self.model = model
        self.iterations = int(iterations)

        # if not isinstance(param_range_min_max, torch.Tensor):
        #    param_range_min_max = torch.Tensor(param_range_min_max).to(smp.device)

        self.param_range_min_max = param_range_min_max
        self.lambda_ = float(lambda_)

    def forward(self, smp, initial, const_: Optional[Tensor] = None):
        cur = initial * 1
        scale = torch.zeros(cur.size(), device='cuda')
        for i in range(self.iterations):
            mu, jac = self.model(cur, const_)
            cur, scale = lm_update(cur, mu, jac, smp, self.lambda_, self.param_range_min_max, scale)

        return cur


def estimate_precision(x, photoncounts, phot_ix, psf_model, param_range, plot_axes=None,
                       axes_scaling=None, axes_units=None, iterations=100, lambda_=50, skip_axes=[],
                       estimator_fn=None, const_=None):
    crlb = []
    prec = []
    rmsd = []
    runtime = []

    mle = torch.jit.script(LM_MLE(psf_model, param_range, iterations, lambda_))

    for i, phot in enumerate(photoncounts):
        x_ = x * 1
        x_[:, phot_ix] = phot
        mu, deriv = psf_model(x_, const_)
        smp = torch.poisson(mu)

        initial = x_ * (torch.rand(x_.shape, device=x.device) * 0.2 + 0.9)

        t0 = time.time()

        if estimator_fn is None:
            estim = mle(smp, initial, const_)
        else:
            estim = estimator_fn(smp, initial)

        errors = x_ - estim

        t1 = time.time()
        runtime.append(len(x) / (t1 - t0))

        prec.append(errors.std(0))
        rmsd.append(torch.sqrt((errors ** 2).mean(0)))

        crlb_i = compute_crlb(mu, deriv, skip_axes=skip_axes)
        crlb.append(crlb_i.mean(0))

    print(runtime)

    crlb = torch.stack(crlb).cpu()
    prec = torch.stack(prec).cpu()
    rmsd = torch.stack(rmsd).cpu()

    if plot_axes is not None:
        figs = []
        for i, ax_ix in enumerate(plot_axes):
            fig, ax = plt.subplots()
            figs.append(fig)
            ax.loglog(photoncounts, axes_scaling[i] * prec[:, ax_ix], label='Precision')
            ax.loglog(photoncounts, axes_scaling[i] * crlb[:, ax_ix], '--', label='CRLB')
            ax.loglog(photoncounts, axes_scaling[i] * rmsd[:, ax_ix], ':k', label='RMSD')
            ax.legend()
            ax.set_title(f'Estimation precision for axis {ax_ix}')
            ax.set_xlabel('Photon count [photons]')
            ax.set_ylabel(f'Precision [{axes_units[i]}]')

        return crlb, prec, rmsd, figs

    return crlb, prec, rmsd


def test_gauss_psf_2D():
    N = 2000
    roisize = 12
    sigma = 1.5
    thetas = np.zeros((N, 4))
    thetas[:, :2] = roisize / 2 + np.random.uniform(-roisize / 8, roisize / 8, size=(N, 2))
    thetas[:, 2] = 1000  # np.random.uniform(200, 2000, size=N)
    thetas[:, 3] = np.random.uniform(1, 10, size=N)

    dev = torch.device('cuda')

    param_range = torch.Tensor([
        [0, roisize - 1],
        [0, roisize - 1],
        [1, 1e9],
        [1e-6, 1e6],
    ]).to(dev)

    thetas = torch.Tensor(thetas).to(dev)
    mu, jac = gauss_psf_2D_fixed_sigma(thetas, roisize, sigma)

    smp = torch.poisson(mu)
    plt.figure()
    plt.imshow(smp[0].cpu().numpy())

    crlb = compute_crlb(mu, jac)

    model = Gaussian2DFixedSigmaPSF(roisize, sigma)
    initial = thetas * (torch.rand(thetas.shape, device=dev) * 0.2 + 0.9)
    mle = LM_MLE(model, lambda_=1e3, iterations=50, param_range_min_max=param_range)
    mle = torch.jit.script(mle)

    for i in range(10):
        t0 = time.time()
        mle(smp, initial, None)
        t1 = time.time()
        print(N / (t1 - t0))

    """
    print(f"CRLB: {crlb.mean(0)} ")
    print(f"Precision: {(estimated - thetas).std(0)}")

    photoncounts = np.logspace(2, 4, 10)
    estimate_precision(thetas, photoncounts, phot_ix=2, 
                       psf_fn = model,
                       plot_axes=[0,1,2],
                       axes_scaling=[100,100,1],
                       axes_units=['nm', 'nm', 'photons'],
                       param_range=param_range)
    """


def make_mod3(depth=0.9, kxy=2, kz=0.01):
    K = 6

    mod = np.array([
        [0, kxy, kz, depth, 0, 1 / K],
        [kxy, 0, kz, depth, 0, 1 / K],
        [0, kxy, kz, depth, 2 * np.pi / 3, 1 / K],
        [kxy, 0, kz, depth, 2 * np.pi / 3, 1 / K],
        [0, kxy, kz, depth, 4 * np.pi / 3, 1 / K],
        [kxy, 0, kz, depth, 4 * np.pi / 3, 1 / K],
    ])
    # mod[:,2]=1
    return torch.Tensor(mod)


# %%


# %%
if __name__ == '__main__':
    test_gauss_psf_2D()

    gauss3D_calib = [
        [1.0573989152908325,
         -0.14864186942577362,
         0.1914146989583969,
         0.10000000149011612],
        [1.0528310537338257,
         0.14878079295158386,
         0.18713828921318054,
         9.999999974752427e-07]]

    np.random.seed(0)

    N = 500
    roisize = 9
    thetas = np.zeros((N, 5))
    thetas[:, :2] = roisize / 2 + np.random.uniform(-roisize / 8, roisize / 8, size=(N, 2))
    thetas[:, 2] = np.random.uniform(-0.3, 0.3, size=N)
    thetas[:, 3] = 1000  # np.random.uniform(200, 2000, size=N)
    thetas[:, 4] = np.random.uniform(1, 10, size=N)

    dev = torch.device('cuda')

    param_range = torch.Tensor([
        [0, roisize - 1],
        [0, roisize - 1],
        [-1, 1],
        [1, 1e9],
        [1e-6, 1e6],
    ]).to(dev)

    thetas = torch.Tensor(thetas).to(dev)
    mu, jac = gauss_psf_2D_astig(thetas, roisize, gauss3D_calib)

    smp = torch.poisson(mu)
    plt.figure()
    plt.imshow(smp[0].cpu().numpy())

    # %%

    psf_model = Gaussian2DAstigmaticPSF(roisize, gauss3D_calib)

    # sf_model = lambda theta: psf_model(sf_model())

    mod = make_mod3().to(dev)

    psf = Gaussian2DAstigmaticPSF(roisize, gauss3D_calib)
    # lm_estimator = torch.jit.script( LM_MLE(psf, param_range, iterations=100, lambda_=1e3) )

    sfi_model = ModulatedIntensitiesModel(bg_factor=20 / len(mod))

    sfi_mu, sfi_jac = sfi_model(thetas, mod[None])
    sfi_crlb = compute_crlb(sfi_mu, sfi_jac, skip_axes=[2])

    sf_mu, sf_jac = SIMFLUXModel(psf).forward(thetas, mod[None])
    sf_crlb = compute_crlb(sf_mu, sf_jac)

    crlb = compute_crlb(mu, jac)
    print(f"SMLM CRLB: {crlb.mean(0)} ")
    print(f"SF CRLB: {sf_crlb.mean(0)} ")

    photoncounts = np.logspace(2, 4, 10)

    sfi_crlb, sfi_prec, sfi_rmsd = estimate_precision(thetas, photoncounts, phot_ix=3,
                                                      psf_model=sfi_model, const_=mod[None],
                                                      param_range=param_range, iterations=100, lambda_=1e3,
                                                      skip_axes=[2])

    astig_crlb, astig_prec, astig_rmsd = estimate_precision(thetas, photoncounts, phot_ix=3,
                                                            psf_model=psf_model, param_range=param_range,
                                                            iterations=120, lambda_=1e3)

    sf_model = torch.jit.script(SIMFLUXModel(psf))
    # simflux_model_3D(theta, mod[None], psf_model, divide_bg=True)

    sf_crlb, sf_prec, sf_rmsd = estimate_precision(thetas, photoncounts, phot_ix=3,
                                                   psf_model=sf_model, const_=mod[None],
                                                   param_range=param_range, iterations=100, lambda_=1e3)

    plot_axes = [0, 1, 2, 3]
    axes_scaling = [100, 100, 1000, 1]
    axes_units = ['nm', 'nm', 'nm', 'photons']

    for i, ax_ix in enumerate(plot_axes):
        fig, ax = plt.subplots()
        # ax.loglog(photoncounts, axes_scaling[i] * astig_prec[:, ax_ix], label='Precision (SF)')
        ax.loglog(photoncounts, axes_scaling[i] * sf_crlb[:, ax_ix], '--b', label='CRLB (SF)')
        ax.loglog(photoncounts, axes_scaling[i] * sf_rmsd[:, ax_ix], 'ob', ms=4, label='RMSD (SF)')
        if ax_ix != 2:
            ...
            ax.loglog(photoncounts, axes_scaling[i] * sfi_crlb[:, ax_ix], '--r', label='CRLB (SFI)')
            ax.loglog(photoncounts, axes_scaling[i] * sfi_rmsd[:, ax_ix], 'or', ms=4, label='RMSD (SFI)')
        ax.loglog(photoncounts, axes_scaling[i] * astig_crlb[:, ax_ix], '--g', label='CRLB (Astig.)')
        ax.loglog(photoncounts, axes_scaling[i] * astig_rmsd[:, ax_ix], 'og', ms=4, label='RMSD (Astig.)')
        # ax.loglog(photoncounts, axes_scaling[i] * astig_prec[:, ax_ix], '+g', ms=4,label='Prec.(Astig.)')
        ax.legend()
        ax.set_title(f'Estimation precision for axis {ax_ix}')
        ax.set_xlabel('Photon count [photons]')
        ax.set_ylabel(f'Precision [{axes_units[i]}]')

    plt.show()

