import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline  # for warping
from transforms3d.axangles import axangle2mat  # for rotation


class dataaugment:
    def __init__(self, data):
        self.data = data

    # 1. Jittering
    # Hyperparameters :  sigma = standard devitation (STD) of the noise
    def DA_Jitter(self, sigma=0.05):
        myNoise = np.random.normal(loc=0, scale=sigma, size=self.data.shape)
        # print(type(myNoise[0]))
        return dataaugment(self.data + myNoise)

    # 2. Scaling
    # Hyperparameters :  sigma = STD of the zoom-in/out factor
    def DA_Scaling(self, sigma=0.1):
        scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, self.data.shape[1]))  # shape=(1,3)
        myNoise = np.matmul(np.ones((self.data.shape[0], 1)), scalingFactor)
        return dataaugment(self.data * myNoise)

    # 3. Magnitude Warping

    # Hyperparameters :  sigma = STD of the random knots for generating curves
    # knot = # of knots for the random curves (complexity of the curves)
    # "Scaling" can be considered as "applying constant noise to the entire samples" whereas "Jittering" can be
    # considered as "applying different noise to each sample".
    # "Magnitude Warping" can be considered as "applying smoothly-varing noise to the entire samples"
    # This example using cubic splice is not the best approach to generate random curves.
    # You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
    def GenerateRandomCurves(self, sigma=0.2, knot=4):
        xx = (np.ones((self.data.shape[1], 1)) * (np.arange(
            0, self.data.shape[0], (self.data.shape[0] - 1) / (knot + 1)))).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, self.data.shape[1]))
        x_range = np.arange(self.data.shape[0])
        r = []
        for i in range(self.data.shape[1]):
            cs_x = CubicSpline(xx[:, i], yy[:, i])
            r.append(cs_x(x_range))
        return np.array(r).transpose()

    def DA_MagWarp(self, sigma, knot):
        return dataaugment(self.data * self.GenerateRandomCurves(sigma, knot))

    # 4. Time Warping
    # Hyperparameters :  sigma = STD of the random knots for generating curves
    # knot = # of knots for the random curves (complexity of the curves)

    def DistortTimesteps(self, sigma=0.2, knot=4):
        tt = self.GenerateRandomCurves(sigma, knot)  # Regard these samples aroun 1 as time intervals
        tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
        # Make the last value to have X.shape[0]
        t_scale = []
        for i in range(self.data.shape[1]):
            m = (self.data.shape[0] - 1) / tt_cum[-1, i]
            t_scale.append(m)
        for j in range(self.data.shape[1]):
            tt_cum[:, j] = tt_cum[:, j] * t_scale[j]
        return tt_cum

    def DA_TimeWarp(self, sigma=0.2, knot=4):
        tt_new = self.DistortTimesteps(sigma, knot)
        X_new = np.zeros(self.data.shape)
        x_range = np.arange(self.data.shape[0])
        for i in range(self.data.shape[1]):
            X_new[:, i] = np.interp(x_range, tt_new[:, i], self.data[:, i])
        return dataaugment(X_new)

    # 5. Rotation
    # Hyperparameters :  N/A

    def DA_Rotation(self):
        axis = np.random.uniform(low=-1, high=1, size=self.data.shape[1])
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        return dataaugment(np.matmul(self.data, axangle2mat(axis, angle)))

    # 6. Permutation
    # Hyperparameters :  nPerm = # of segments to permute
    # minSegLength = allowable minimum length for each segment

    def DA_Permutation(self, nPerm=4, minSegLength=10):
        X_new = np.zeros(self.data.shape)
        idx = np.random.permutation(nPerm)
        bWhile = True
        while bWhile == True:
            segs = np.zeros(nPerm + 1, dtype=int)
            segs[1:-1] = np.sort(np.random.randint(minSegLength, self.data.shape[0] - minSegLength, nPerm - 1))
            segs[-1] = self.data.shape[0]
            if np.min(segs[1:] - segs[0:-1]) > minSegLength:
                bWhile = False
        pp = 0
        for ii in range(nPerm):
            x_temp = self.data[segs[idx[ii]]:segs[idx[ii] + 1], :]
            X_new[pp:pp + len(x_temp), :] = x_temp
            pp += len(x_temp)
        return (X_new)
