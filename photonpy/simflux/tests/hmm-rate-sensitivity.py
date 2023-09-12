"""
Test frame selection on a single emitter with two-state blinking.


"""
import numpy as np
import math
import scipy.special as sps

import sys

sys.path.append("..")

import smlmlib.gaussian
import smlmlib.silm
import smlmlib.util as su

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys
import os

from scipy.stats import poisson


if not os.path.exists("pdfs"):
    os.makedirs("pdfs")

dbg = True
gaussian = smlmlib.gaussian.Gaussian(debugMode=dbg)
silm = smlmlib.silm.SILM(debugMode=dbg)
useCuda = False

# Global constants
patternDepth = 0.9
NA = 1.2
imgw = 10
wavelength = 0.532
pixelsize = 0.1
sigma = 0.42 * wavelength / (2 * NA) / pixelsize
minIllumL = wavelength / (2 * NA) / pixelsize
omega = 2 * math.pi / minIllumL

np.set_printoptions(precision=4)
axesNames = ["x", "y", "I", "bg"]
axesUnits = ["nm", "nm", "photons", "photons/pixel"]
axesFactors = [1000 * pixelsize, 1000 * pixelsize, 1, 1]


def PrintNanTraces(msg, results):
    for i in range(len(results)):
        r = results[i]
        if math.isnan(np.sum(r.estimate)):
            print(i)


def ComputeHMMParams(T_on, T_off, fps, intensityThreshold, p_on_treshold=0.9, excitationThreshold=0.05):
    p_on = T_on / (T_on + T_off)  # prior
    k_on2off = 1 / T_on / fps
    k_off2on = 1 / T_off / fps
    # this is not entirely accurate but ok for now.
    # In reality, the continuous time state can switch multiple times within one discrete-time step
    p_on2off = poisson.pmf(1, k_on2off)
    p_off2on = poisson.pmf(1, k_off2on)
    print(f"On2off prob: {p_on2off}. Off2on: {p_off2on}")
    return smlmlib.silm.BlinkHMM_Params.make(
        p_on, p_on2off, p_off2on, p_on_treshold, intensityThreshold, excitationThreshold
    )


def ComputeBlinkHMMExpectedValue(phi_n, theta, numEPP, silm_p, blinkStartTime):
    numBgFrames = 2
    numspots = len(theta)

    startTimeRange = numEPP

    blinkStartTime = numBgFrames + startTimeRange * np.ones(numspots)
    blinkEndTime = blinkStartTime + numEPP

    nframes = numEPP + startTimeRange + numBgFrames * 2 + 1
    mu = np.zeros((numspots, nframes, silm_p.imgw, silm_p.imgw))
    frameTheta = np.zeros((numspots, nframes, 4))
    for f in range(nframes):
        # frame 1, startTime = 1.4: 1+1-1.4 = 0.6
        endFraction = np.clip(blinkEndTime - f, 0, 1)
        startFraction = np.clip(f + 1 - blinkStartTime, 0, 1)
        ontimes = endFraction * startFraction

        #       print(f"f={f}, startFrac={startFraction} endFRaction={endFraction}")
        #       print(f"f={f}, ontimes={ontimes}")
        thetaAdj = theta * 1
        thetaAdj[:, 2] *= ontimes
        frameTheta[:, f] = thetaAdj
        frame_mu, frame_fi = silm.SILM_ASW_ComputeFisherMatrix(phi_n, thetaAdj, 1, f % numEPP, silm_p)
        mu[:, f] = frame_mu[:, 0]

    return mu


def plot_estimates(estimates, estimate, total_crlb, title):
    fig, ax = plt.subplots()
    ax.scatter(estimates[:, 0], estimates[:, 1], label=f"Individual fits")
    ax.scatter(estimate[0], estimate[1], label=f"Combined estimate")
    ax.add_patch(Ellipse((estimate[0], estimate[1]), total_crlb[0], total_crlb[1], color="r", fill=False))
    plt.xlabel("x [pixels]")
    plt.ylabel("y [pixels]")
    plt.legend()
    plt.title(title)
    return fig


def rate_sensitivity_plot(prefix="pdfs", n=1000):
    if silm.debugMode:
        n = 50

    nTheta = 4  # num parameters
    numEPP = 6
    rates = np.linspace(0.001, 0.01, 10)
    nrates = len(rates)

    errors = np.zeros((nrates, n, nTheta))
    errors_g = np.zeros((nrates, n, nTheta))
    errors_blink_rs = np.zeros((nrates, n, nTheta))
    crlb = np.zeros((nrates, nTheta))
    crlb_g = np.zeros((nrates, nTheta))

    np.random.seed(0)

    photons = 2000
    silm_p = silm.Params(imgw, numEPP, sigma, patternDepth, omega, levMarIt=100, startLambdaStep=0.1)
    phi = np.tile(np.arange(numEPP / 2) * 4 * math.pi / numEPP, 2)
    phi_n = np.tile(phi, (n, 1))
    center = [imgw / 2, imgw / 2, photons, 6]
    theta = (
        np.tile(center, (n, 1)) + np.random.uniform([-1, -1, 0, 0], [1, 1, 0, 0], size=(n, 4)) * minIllumL / 2
    )  #  move xy in a range of 2 pixels

    mu, fi = silm.SILM_ASW_ComputeFisherMatrix(phi_n, theta, numEPP, 0, silm_p)

    # Blink at random start position. HMM should reject the frames that might not be fully on
    mu_blink_rs = ComputeBlinkHMMExpectedValue(phi_n, theta, numEPP, silm_p, np.random.uniform(size=n))
    smp_blink_rs = np.array(np.random.poisson(mu_blink_rs), dtype=np.float32)

    mu_g, fi_g = gaussian.ComputeFisherMatrix(theta, sigma, imgw)
    crlb = np.mean(su.crlb(fi), 0)
    crlb_g = np.mean(su.crlb(fi_g), 0)

    for i, onrate in enumerate(rates):
        hmmParams = ComputeHMMParams(
            T_on=onrate, T_off=onrate * 10, fps=400, intensityThreshold=photons / 3, p_on_treshold=0.8
        )
        estim_blink_rs, results_blink_rs = silm.BlinkHMM_ComputeMLE(
            smp_blink_rs, phi_n, silm_p, 0, hmmParams, cuda=useCuda
        )
        errors_blink_rs[i] = estim_blink_rs - theta
        PrintNanTraces(f"Nan indices for {photons} photons and {numEPP} EPP\n", results_blink_rs)

    def makePlot(axis):
        axn = axesNames[axis]
        stdev = np.std(errors, 2)
        stdev_g = np.std(errors_g, 2)
        stdev_blink_rs = np.std(errors_blink_rs, 2)

        fig = plt.figure()

        for i in range(NK):
            plt.plot(
                photons,
                axesFactors[axis] * stdev_blink[:, i, axis],
                "d-",
                label="(SILM-Blink) MLE K={0}".format(N_EIP[i]),
            )
            plt.plot(
                photons,
                axesFactors[axis] * stdev_blink_rs[:, i, axis],
                "x-",
                label="(SILM-Blink-RS) MLE K={0}".format(N_EIP[i]),
            )

        plt.plot(
            [rates[0], rates[-1]],
            axesFactors[0] * crlb[:, i, 0],
            label="(SILM) CRLB K={0}".format(N_EIP[i]),
            linestyle=":",
        )

        #        for i in range(NK):
        plt.plot(photons, axesFactors[axis] * stdev_g[:, 0, axis], "+-", label="(G2D) MLE")
        plt.plot(photons, axesFactors[axis] * crlb_g[:, axis], label="(G2D) CRLB", linestyle=":")

        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True)
        plt.xlabel("$\\theta_I$ [photons]")
        plt.ylabel("Std. Deviation of errors in {0} [{1}]".format(axn, axesUnits[axis]))
        plt.title(f"MLE precision for different localization methods ({n} samples per pt)")
        plt.legend(loc=1)
        plt.show()
        fig.savefig("pdfs/{0}mle-stdev-photons-k-{1}.pdf".format(prefix, axn), bbox_inches="tight")

        uncertaintyFactorCRLB = np.mean(crlb_g[:, axis] / crlb[:, 0, axis])
        uncertaintyFactorMLE = np.mean(stdev_g[:, 0, axis] / stdev[:, 0, axis])
        print(
            "SILM improvement in uncertainy for {0} axis: MLE {1}. CRLB: {2}".format(
                axn, uncertaintyFactorMLE, uncertaintyFactorCRLB
            )
        )


rate_sensitivity_plot()
