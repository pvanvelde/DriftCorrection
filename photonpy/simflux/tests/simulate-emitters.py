"""
3 states: 0=On, 1=Off, 2=Bleached.  
Kinetic parameters: [ k01, k10, k12 ]. 
Note that these are not per frame, but per received unit of irradiance. 
The irradiance per frame is between 0 and 1, based on the position of the emitter.

To account for the excitation patterns, we use 'irradiance-time' instead of just regular time. 
At the start, randomly pick the amount of irradiance 'qbudget' that will switch it to another state.



State model:

   <k10<     k12>
On ----- Off ---- Bleached
   >k01>     
   
k10 and k01 are rates for Exponential distribution. 
k12 is the probability of bleaching every time the molecule switches to off state.

Recommended to use spyder to run this code:
INSTALL:
pip install PyQt5==5.9.2
pip install scipy matplotlib numpy
pip install spyder
spyder3

"""
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import copy

import smlmlib.gaussian
import smlmlib.silm
import smlmlib.util as su
from enum import IntEnum

dbg = False
gaussian = smlmlib.gaussian.Gaussian(debugMode=dbg)
silm = smlmlib.silm.SILM(debugMode=dbg)

patternDepth = 1
NA = 1.2
imgw = 10
fovsize = 256
wavelength = 0.532
pixelsize = 0.1
sigma = 0.42 * wavelength / (2 * NA) / pixelsize
minIllumL = wavelength / (2 * NA) / pixelsize
omega = 2 * math.pi / minIllumL
silmParams = silm.Params(imgw, 6, sigma, patternDepth=1, omega=omega)


class EmitterState(IntEnum):
    ON = 0
    OFF = 1
    DEAD = 2


class Kinetics:
    def __init__(self, k_on=0.1, k_off=0.5, p_bl=0.4):
        self.k_on = k_on
        self.k_off = k_off
        self.p_bl = p_bl


class Emitter:
    def __init__(self, pos):
        self.remaining = 0  # remaining time in current state
        self.state = EmitterState.OFF
        self.position = pos
        self.currentFrameOnTime = 0
        self.switches = 0

    def advanceTime(self, dq, kinetics: Kinetics):
        self.currentFrameOnTime = 0

        if self.state == EmitterState.DEAD:
            return 0

        onTime = 0
        while self.remaining <= dq:
            self.switches += 1
            dq -= self.remaining
            if self.state == EmitterState.ON:
                onTime += self.remaining
                if np.random.binomial(1, kinetics.p_bl):
                    self.state = EmitterState.DEAD
                else:
                    self.state = EmitterState.OFF
                    self.remaining = np.random.exponential(kinetics.k_on)
            #                    print('switching off for {0} qt', self.remaining)
            else:
                self.remaining = np.random.exponential(kinetics.k_off)
                self.state = EmitterState.ON
        #                print('switching on for {0} qt', self.remaining)

        self.remaining -= dq
        if self.state == EmitterState.ON:
            onTime += dq

        self.currentFrameOnTime = onTime


def advance_time(dt, emitters, k_rates, excitationPattern):
    for i in range(len(emitters)):
        e = emitters[i]
        q = excitationPattern(e.position) * dt
        e.advanceTime(q, k_rates)


def draw_emitters(imgw, emitters, sigma):
    image = np.zeros([imgw, imgw], dtype=np.float32)
    gaussian.Gauss2D_Draw(image, [[e.position[0], e.position[1], sigma, sigma, e.currentFrameOnTime] for e in emitters])
    return image


def generate_emitters(imgw, numEmitters, kinetics: Kinetics):
    emitters = []
    positions = np.random.uniform([0, 0, 0, 1000], [imgw - 1, imgw - 1, 0, 1000], [numEmitters, 4])
    for i in range(numEmitters):
        e = Emitter(positions[i])
        p_on = kinetics.k_on / (kinetics.k_on + kinetics.k_off)
        if np.random.binomial(1, p_on):
            e.state = EmitterState.ON
        emitters.append(e)
    return emitters


def simulate_epi(emitters, fps, nframes, kinetics, silmParams):
    emitters = copy.deepcopy(emitters)
    dt = 1 / fps
    images = []
    mot = np.zeros((nframes, 2))
    for f in range(nframes):
        advance_time(dt, emitters, kinetics, lambda xyz: 1)
        meanOnTime = np.mean([e.currentFrameOnTime for e in emitters])

        image = draw_emitters(fovsize, emitters, silmParams.sigma)
        images.append(image)
        mot[f] = [f * dt, meanOnTime]

        if f % 10 == 0:
            sys.stdout.write(".")

    return images, mot


def simulate_silm(emitters, fps, nframes, kinetics, silmParams):
    emitters = copy.deepcopy(emitters)
    phi = np.tile(np.arange(silmParams.numepp / 2) * 4 * math.pi / silmParams.numepp, 2)
    images = []
    mot = np.zeros((nframes, 2))
    dt = 1 / fps

    silm_state = 0
    for f in range(nframes):
        silm_state = (silm_state + 1) % silmParams.numepp

        def computeExcitationIntensity(xyz):
            return su.excitation_intensity(silmParams, phi, silm_state, xyz[0], xyz[1])

        advance_time(dt, emitters, kinetics, computeExcitationIntensity)
        meanOnTime = np.mean([e.currentFrameOnTime for e in emitters])
        #        print(f"frame {f}: mean on time: {meanOnTime}")

        image = draw_emitters(imgw, emitters, silmParams.sigma)
        images.append(image)

        mot[f] = [f * dt, meanOnTime]
        if f % 10 == 0:
            sys.stdout.write(".")

    sys.stdout.write("\n")
    return images, mot


def test_advance_time():
    plt.figure()
    simtime = 10
    kinetics = Kinetics()
    emlist1 = generate_emitters(fovsize, 2000, kinetics)
    for fps in [10, 20, 40, 100]:
        emlist = copy.deepcopy(emlist1)
        tot = []
        sw = []
        nframes = fps * simtime
        print(f"FPS={fps}: #{nframes} frames")
        for f in range(nframes):
            advance_time(1 / fps, emlist, kinetics, lambda xyz: 1)

            tot.append([f / fps, np.sum([e.currentFrameOnTime for e in emlist])])
            sw.append([f / fps, np.sum([e.switches for e in emlist])])

        tot = np.array(tot)
        sw = np.array(sw)
        #        plt.plot(tot[:,0], np.cumsum(tot[:,1]), label=f"fps={fps}" )
        #      plt.plot(tot[:,0],tot[:,1],label=f"fps={fps}")
        plt.plot(sw[:, 0], sw[:, 1], label=f"fps={fps}")
        plt.legend()


def test_movie():
    kinetics = Kinetics()
    emlist = generate_emitters(fovsize, 1000, kinetics)

    img_epi, mot_epi = simulate_epi(emlist, 100, 100, kinetics, silmParams)
    su.save_movie(img_epi, "epi.mp4")


def test():
    img_silm, mot_silm = simulate_silm(
        emlist, 100 * silmParams.numepp, 100 * silmParams.numepp, kinetics, silmParams, show_some_frames
    )

    plt.figure()
    plt.xlabel("Time [s]")
    plt.plot(mot_epi[:, 0], mot_epi[:, 1], label="Epifluorescence")
    plt.plot(mot_silm[:, 0], mot_silm[:, 1], label="SILM")
    plt.legend()


test_advance_time()
# test_movie()
