import numpy as np
import math

class UKF:
    def __init__(self):
        self.c1 = 1
        self.c2 = 12
        self.c3 = 7
        self.R = 0.3
        self.Q = 0.3
        self.w = np.random.normal(0,2**2)
        self.r = np.random.normal(0,2**2)
        self.sensor_noise = np.random.normal(0.0, 0.2**2)
        self.k = 3
        self.alpha = 0.5
        self.beta = 2
        n = 1
        self.gamma = (self.alpha ** 2)*(n +self.k)-n
        self.weight_avg0 = self.gamma / (n + self.gamma)
        self.weight_cov0 = self.weight_avg0 + (1 - (self.gamma ** 2) + self.beta)
        self.weight_avg1 = self.weight_avg2 = self.weight_cov1 = self.weight_cov2 = 1 / (2 * (n + self.gamma))


    def make_data(self, old_mean, old_cov):
        measure_x = []
        sigma_list = []
        for k in range(100):
            sigma = old_mean
            sigma2 = old_mean + self.gamma*np.math.sqrt(old_cov)
            sigma3 = old_mean - self.gamma*np.math.sqrt(old_cov)
            sigma_point = [sigma, sigma2, sigma3]
            sig1 = self.c1 * old_mean + ((self.c2 * old_mean) / (1 + old_mean ** 2)) + self.c3 * math.cos(1.2 * (k - 1)) + self.w
            sig2 = self.c1 * old_mean + ((self.c2 * old_mean) / (1 + old_mean ** 2)) + self.c3 * math.cos(1.2 * (k - 1)) + self.w
            x = self.c1*old_mean+((self.c2*old_mean)/(1+old_mean**2))+self.c3*math.cos(1.2*(k - 1))+self.w

            sigma_list.append([sigma,sigma2,sigma3])
            old_mean = x
            measure_x.append(x)

        return measure_x, sigma_list

    def Uscented_kalman(self, old_mean, old_cov):
        sigma =


if __name__=='__main__':
    ukf = UKF()
    measurement, sigma_list = ukf.make_data()
    print(measurement)