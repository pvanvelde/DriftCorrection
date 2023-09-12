# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 23:20:06 2021

@author: jelmer
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from typing import List, Tuple
import scipy.special

from numba import njit

cat = np.concatenate
sqrt = np.sqrt
rand = np.random.uniform
exp = np.exp
erf = scipy.special.erf


def gauss_psf_2D(theta, numpixels: int):
    """
    theta: [x,y,N,bg,sigma].T
    """
    pi = 3.141592653589793  # torch needs to include pi

    x, y, N, bg, sigma = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3], theta[:, 4]
    pixelpos = np.arange(0, numpixels)

    OneOverSqrt2PiSigma = (1.0 / (sqrt(2 * pi) * sigma))[:, None, None]
    OneOverSqrt2Sigma = (1.0 / (sqrt(2) * sigma))[:, None, None]

    # Pixel centers
    Xc = pixelpos[None, None, :]
    Yc = pixelpos[None, :, None]
    Xexp0 = (Xc - x[:, None, None] + 0.5) * OneOverSqrt2Sigma
    Xexp1 = (Xc - x[:, None, None] - 0.5) * OneOverSqrt2Sigma
    Ex = 0.5 * erf(Xexp0) - 0.5 * erf(Xexp1)
    dEx = OneOverSqrt2PiSigma * (exp(-Xexp1 ** 2) - exp(-Xexp0 ** 2))
    dEx_dSigma = (exp(-Xexp1 ** 2) * Xexp1 - exp(-Xexp0 ** 2) * Xexp0) / sqrt(pi)

    Yexp0 = (Yc - y[:, None, None] + 0.5) * OneOverSqrt2Sigma
    Yexp1 = (Yc - y[:, None, None] - 0.5) * OneOverSqrt2Sigma
    Ey = 0.5 * erf(Yexp0) - 0.5 * erf(Yexp1)
    dEy = OneOverSqrt2PiSigma * (exp(-Yexp1 ** 2) - exp(-Yexp0 ** 2))
    dEy_dSigma = (exp(-Yexp1 ** 2) * Yexp1 - exp(-Yexp0 ** 2) * Yexp0) / sqrt(pi)

    mu = N[:, None, None] * Ex * Ey + bg[:, None, None]
    dmu_x = N[:, None, None] * Ey * dEx
    dmu_y = N[:, None, None] * Ex * dEy
    dmu_I = Ex * Ey
    dmu_bg = 1 + mu * 0
    dmu_sigma = N[:, None, None] * (Ex * dEy_dSigma + dEx_dSigma * Ey)

    deriv = np.stack((dmu_x, dmu_y, dmu_I, dmu_bg, dmu_sigma), -1)
    return mu, deriv


def gauss_psf_2D_fixed_sigma(theta, roisize: int, sigma: float):
    sigma_ = np.ones((len(theta), 1)) * sigma
    theta_ = cat((theta, sigma_), -1)

    mu, jac = gauss_psf_2D(theta_, roisize)
    return mu, jac[..., :-1]


def gauss_psf_2D_astig(theta, numpixels: int, calib: List[List[float]]):
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
    sigma_x = s0_x * sqrt(1 + tzx ** 2 / d_x ** 2 + A_x * tzx ** 3 / d_x ** 3)
    sigma_y = s0_y * sqrt(1 + tzy ** 2 / d_y ** 2 + A_y * tzy ** 3 / d_y ** 3)

    OneOverSqrt2PiSigma_x = 1 / (sqrt2pi * sigma_x)
    OneOverSqrt2Sigma_x = 1 / (sqrt2 * sigma_x)
    OneOverSqrt2PiSigma_y = 1 / (sqrt2pi * sigma_y)
    OneOverSqrt2Sigma_y = 1 / (sqrt2 * sigma_y)

    pixelpos = np.arange(0, numpixels)
    Xc = pixelpos[None, None, :]
    Yc = pixelpos[None, :, None]

    # Pixel centers
    Xexp0 = (Xc - tx + 0.5) * OneOverSqrt2Sigma_x
    Xexp1 = (Xc - tx - 0.5) * OneOverSqrt2Sigma_x
    Ex = 0.5 * erf(Xexp0) - 0.5 * erf(Xexp1)
    dEx = OneOverSqrt2PiSigma_x * (exp(-Xexp1 ** 2) - exp(-Xexp0 ** 2))

    Yexp0 = (Yc - ty + 0.5) * OneOverSqrt2Sigma_y
    Yexp1 = (Yc - ty - 0.5) * OneOverSqrt2Sigma_y
    Ey = 0.5 * erf(Yexp0) - 0.5 * erf(Yexp1)
    dEy = OneOverSqrt2PiSigma_y * (exp(-Yexp1 ** 2) - exp(-Yexp0 ** 2))

    G21y = 1 / (sqrt2pi * sigma_y * sigma_y) * (
            (Yc - ty - 0.5) * exp(-(Yc - ty - 0.5) * (Yc - ty - 0.5) / (2.0 * sigma_y * sigma_y)) -
            (Yc - ty + 0.5) * exp(-(Yc - ty + 0.5) * (Yc - ty + 0.5) / (2.0 * sigma_y * sigma_y)))

    mu = tbg + tI * Ex * Ey
    dmu_dx = tI * dEx * Ey
    dmu_dy = tI * dEy * Ex

    G21x = 1 / (sqrt2pi * sigma_x * sigma_x) * (
            (Xc - tx - 0.5) * exp(-(Xc - tx - 0.5) * (Xc - tx - 0.5) / (2 * sigma_x * sigma_x)) -
            (Xc - tx + 0.5) * exp(-(Xc - tx + 0.5) * (Xc - tx + 0.5) / (2 * sigma_x * sigma_x)))

    dMuSigmaX = tI * Ey * G21x
    dMuSigmaY = tI * Ex * G21y

    dSigmaXThetaZ = (s0_x * (2 * tzx / d_x ** 2 + A_x * 3 * tzx ** 2 / d_x ** 2) /
                     (2 * sqrt(1 + tzx ** 2 / d_x ** 2 + A_x * tzx ** 3 / d_x ** 3)))
    dSigmaYThetaZ = (s0_y * (2 * tzy / d_y ** 2 + A_y * 3 * tzy ** 2 / d_y ** 2) /
                     (2 * sqrt(1 + tzy ** 2 / d_y ** 2 + A_y * tzy ** 3 / d_y ** 3)))

    dmu_dz = dMuSigmaX * dSigmaXThetaZ + dMuSigmaY * dSigmaYThetaZ

    dmu_dI0 = Ex * Ey
    dmu_dIbg = dmu_dx * 0 + 1

    return mu, np.stack((dmu_dx, dmu_dy, dmu_dz, dmu_dI0, dmu_dIbg), -1)


def sf_modulation(xyz, mod):
    k = mod[:, :, :3]
    phase = mod[:, :, 4]
    depth = mod[:, :, 3]
    relint = mod[:, :, 5]
    # z is ignored
    em_phase = (xyz[:, None] * k).sum(-1) - phase

    deriv = depth[:, :, None] * k * np.cos(em_phase)[:, :, None] * relint[:, :, None]  # [spots,patterns,coords]
    intensity = (1 + depth * np.sin(em_phase)) * relint

    return intensity, deriv


def simflux_model_3D(params, mod, psf_model, divide_bg=False):
    """
    params: [batchsize, xyzIb]
    mod: [batchsize, frame_window, 6] . format: kx,ky,kz,depth,phase,relint
    """

    # mu = bg + I * P(xyz) * PSF(xyz)
    xyz = params[:, :3]
    I = params[:, [3]]
    bg = params[:, [4]]

    if divide_bg:
        bg /= mod.shape[1]

    mod_intensity, mod_deriv = sf_modulation(xyz, mod)
    normalized_psf_params = cat((xyz, np.ones(I.shape), np.zeros(I.shape)), -1)
    psf_ev, psf_deriv = psf_model(normalized_psf_params)

    phot = I[:, None, None]
    mu = bg[:, None, None] + mod_intensity[:, :, None, None] * phot * psf_ev[:, None, :, :]

    deriv_xyz = phot[..., None] * (mod_deriv[:, :, None, None] * psf_ev[:, None, ..., None] +
                                   mod_intensity[:, :, None, None, None] * psf_deriv[:, None, ..., :3])

    deriv_I = psf_ev[:, None] * mod_intensity[:, :, None, None]

    deriv_bg = np.ones(deriv_I.shape)

    return mu, cat((deriv_xyz,
                    deriv_I[..., None],
                    deriv_bg[..., None]), -1)


def simflux_model_intensities(params, mod, bg_factor: float = 1):
    """
    params: [batchsize, ...]
    mod: [batchsize, frame_window, 6] . format: kx,ky,kz,depth,phase,relint

    basically simflux_model but with all psf derivatives set to zero, so they don't contribute any FI.

    psf_ev is now just 1.
    This brings up a problem for the background, which is defined per pixel.
    As a hack, bg_factor can be set to the number of pixels that the PSF roughly covers,
    divided by number of patterns (as in normal simflux)
    """
    xyz = params[:, :3]
    I = params[:, [3]]
    bg = params[:, [4]] * bg_factor

    mod_intensity, mod_deriv = sf_modulation(xyz, mod)

    mu = bg + mod_intensity * I

    deriv_xyz = I[..., None] * mod_deriv
    deriv_I = mod_intensity
    deriv_bg = np.ones(deriv_I.shape)

    return mu, cat((deriv_xyz,
                    deriv_I[..., None],
                    deriv_bg[..., None]), -1)


def compute_crlb(mu, jac, *, skip_axes: List[int] = []):
    """
    Compute crlb from expected value and per pixel derivatives.
    mu: [N, H, W]
    jac: [N, H,W, coords]
    """
    naxes = jac.shape[-1]
    axes = [i for i in range(naxes) if not i in skip_axes]
    jac = jac[..., axes]

    sample_dims = tuple(np.arange(1, len(mu.shape)))

    fisher = np.matmul(jac[..., None], jac[..., None, :])  # derivative contribution
    fisher = fisher / mu[..., None, None]  # px value contribution
    fisher = fisher.sum(sample_dims)

    crlb = np.zeros((len(mu), naxes))
    crlb[:, axes] = np.sqrt(np.diagonal(np.linalg.inv(fisher), axis1=1, axis2=2))
    return crlb


def lm_alphabeta(mu, jac, smp):
    """
    mu: [batchsize, numsamples]
    jac: [batchsize, numsamples, numparams]
    smp: [batchsize, numsamples]
    """
    # assert np.array_equal(smp.shape, mu.shape)
    sampledims = tuple(i for i in range(1, len(smp.shape)))

    invmu = 1.0 / np.maximum(mu, 1e-9)
    af = smp * invmu ** 2

    jacm = np.matmul(jac[..., None], jac[..., None, :])
    alpha = jacm * af[..., None, None]
    alpha = alpha.sum(sampledims)

    beta = (jac * (smp * invmu - 1)[..., None]).sum(sampledims)
    return alpha, beta


def lm_update(cur, mu, jac, smp, lambda_: float, param_range_min_max):
    """
    Separate some of the calculations to speed up with jit script
    """
    alpha, beta = lm_alphabeta(mu, jac, smp)

    K = cur.shape[-1]

    if True:  # scale invariant. Helps when parameter scales are quite different
        # For a matrix A, (element wise A*A).sum(0) is the same as diag(A^T * A)
        scale = (alpha * alpha).sum(1)
        scale /= scale.mean(1, keepdims=True)  # normalize so lambda scale is not model dependent
        alpha += lambda_ * scale[:, :, None] * np.identity(K)[None]
    else:
        # regular LM, non scale invariant
        alpha += lambda_ * np.identity(K)[None]

    steps = np.linalg.solve(alpha, beta)
    cur = cur + steps
    cur = np.maximum(cur, param_range_min_max[None, :, 0])
    cur = np.minimum(cur, param_range_min_max[None, :, 1])
    return cur


def lm_mle(model, initial, smp, param_range_min_max, iterations=50, lambda_=1, store_traces=False):
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
    cur = initial * 1

    traces = [cur]

    assert len(smp) == len(initial)

    for i in range(iterations):
        mu, jac = model(cur)

        cur = lm_update(cur, mu, jac, smp, lambda_, param_range_min_max)

        if store_traces:
            traces.append(cur)

    if store_traces:
        return cur, np.stack(traces, -1)

    return cur


def estimate_precision(x, photoncounts, phot_ix, psf_fn, param_range, plot_axes=None,
                       axes_scaling=None, axes_units=None, iterations=100, lambda_=50, skip_axes=[],
                       estimator_fn=None):
    crlb = []
    prec = []
    rmsd = []
    runtime = []

    for i, phot in enumerate(photoncounts):
        x_ = x * 1
        x_[:, phot_ix] = phot
        mu, deriv = psf_fn(x_)
        smp = np.random.poisson(mu)

        initial = x_ * (np.random.uniform(size=x_.shape) * 0.2 + 0.9)

        t0 = time.time()

        if estimator_fn is None:
            estim = lm_mle(psf_fn, initial, smp, param_range_min_max=param_range,
                           iterations=iterations, lambda_=lambda_, store_traces=False)
        else:
            estim = estimator_fn(initial, smp)

        errors = x_ - estim

        t1 = time.time()
        runtime.append(len(x) / (t1 - t0))

        prec.append(errors.std(0))
        rmsd.append(np.sqrt((errors ** 2).mean(0)))

        crlb_i = compute_crlb(mu, deriv, skip_axes=skip_axes)
        crlb.append(crlb_i.mean(0))

    print(runtime)

    crlb = np.stack(crlb)
    prec = np.stack(prec)
    rmsd = np.stack(rmsd)

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
    N = 1000
    roisize = 12
    sigma = 1.5
    thetas = np.zeros((N, 4))
    thetas[:, :2] = roisize / 2 + np.random.uniform(-roisize / 8, roisize / 8, size=(N, 2))
    thetas[:, 2] = 1000  # np.random.uniform(200, 2000, size=N)
    thetas[:, 3] = np.random.uniform(1, 10, size=N)

    param_range = [
        [0, roisize - 1],
        [0, roisize - 1],
        [1, 1e9],
        [1e-6, 1e6],
    ]

    mu, jac = gauss_psf_2D_fixed_sigma(thetas, roisize, sigma)

    smp = np.poisson(mu)
    plt.figure()
    plt.imshow(smp[0])

    crlb = compute_crlb(mu, jac)

    model = lambda theta: gauss_psf_2D_fixed_sigma(theta, roisize, sigma)
    initial = thetas * (np.random.uniform(size=thetas.shape) * 0.2 + 0.9)
    estimated = lm_mle(model, initial, smp, lambda_=1, iterations=20, param_range_min_max=param_range)

    print(f"CRLB: {crlb.mean(0)} ")
    print(f"Precision: {(estimated - thetas).std(0)}")

    photoncounts = np.logspace(2, 4, 10)

    estimate_precision(thetas, photoncounts, phot_ix=2,
                       psf_fn=model,
                       plot_axes=[0, 1, 2],
                       axes_scaling=[100, 100, 1],
                       axes_units=['nm', 'nm', 'photons'],
                       param_range=param_range)


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
    return mod


# %%

# test_gauss_psf_2D()

# %%
if __name__ == '__main__':

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

    N = 50
    roisize = 9
    thetas = np.zeros((N, 5))
    thetas[:, :2] = roisize / 2 + np.random.uniform(-roisize / 8, roisize / 8, size=(N, 2))
    thetas[:, 2] = np.random.uniform(-0.3, 0.3, size=N)
    thetas[:, 3] = 1000  # np.random.uniform(200, 2000, size=N)
    thetas[:, 4] = np.random.uniform(1, 10, size=N)

    param_range = np.array([
        [0, roisize - 1],
        [0, roisize - 1],
        [-1, 1],
        [1, 1e9],
        [1e-6, 1e6],
    ])

    mu, jac = gauss_psf_2D_astig(thetas, roisize, gauss3D_calib)

    smp = np.random.poisson(mu)
    plt.figure()
    plt.imshow(smp[0])

    # %%
    psf_model = lambda theta: gauss_psf_2D_astig(theta, roisize, gauss3D_calib)

    # sf_model = lambda theta: psf_model(sf_model())

    mod = make_mod3()

    sfi_mu, sfi_jac = simflux_model_intensities(thetas, mod[None])
    sfi_crlb = compute_crlb(sfi_mu, sfi_jac, skip_axes=[2])

    sf_mu, sf_jac = simflux_model_3D(thetas, mod[None], psf_model)
    sf_crlb = compute_crlb(sf_mu, sf_jac)

    crlb = compute_crlb(mu, jac)
    print(f"SMLM CRLB: {crlb.mean(0)} ")
    print(f"SF CRLB: {sf_crlb.mean(0)} ")

    photoncounts = np.logspace(2, 4, 10)

    sfi_model = lambda theta: simflux_model_intensities(theta, mod[None], bg_factor=20 / len(mod))
    sfi_crlb, sfi_prec, sfi_rmsd = estimate_precision(thetas, photoncounts, phot_ix=3,
                                                      psf_fn=sfi_model,
                                                      param_range=param_range, iterations=100, lambda_=1e3,
                                                      skip_axes=[2])

    astig_crlb, astig_prec, astig_rmsd = estimate_precision(thetas, photoncounts, phot_ix=3,
                                                            psf_fn=psf_model, param_range=param_range,
                                                            iterations=120, lambda_=1e3)

    sf_model = lambda theta: simflux_model_3D(theta, mod[None], psf_model, divide_bg=True)
    sf_crlb, sf_prec, sf_rmsd = estimate_precision(thetas, photoncounts, phot_ix=3,
                                                   psf_fn=sf_model,
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

