import numpy as np
import math
import matplotlib.pyplot as plt
c1 = 1
c2 = 12
c3 = 7

class Kalman:
    def __init__(self):
        self.c1 = 1
        self.c2 = 12
        self.c3 = 7
        self.R = 0.3
        self.Q = 0.3
        self.w = np.random.normal(0,2**2)
        self.r = np.random.normal(0,2**2)
        self.sensor_noise = np.random.normal(0.0, 0.2**2)

    def make_data(self):
        old_x = 1
        measure_x = []
        for k in range(100):
            x = self.c1*old_x+((self.c2*old_x)/(1+old_x**2))+self.c3*math.cos(1.2*(k - 1))+self.w
            old_x = x
            measure_x.append(x)

        return measure_x

    def Extended_kalman(self,old_mean,old_cov, k,measure):
        jac = self.c1 + self.c2*((1 - measure[k] ** 2) / (1 + measure[k] ** 2))
        jac_T = np.transpose(jac)

        H = measure[k]/ (10)
        H_T = np.transpose(H)
        z = ((measure[k] ** 2)/20) + self.sensor_noise

        mean_p = self.c1*old_mean + ((self.c2*old_mean)/(1+old_mean**2)) + self.c3*math.cos(1.2*(k - 1))
        h = (mean_p ** 2) / (20)



        cov_p = jac*old_cov*jac_T + self.R
        kg = cov_p*H_T*(1/(H*cov_p*H_T+self.Q))
        mean = mean_p + kg*(z-h)
        cov = (1-kg*H)*cov_p

        return mean, cov


if __name__=='__main__':
    kalman = Kalman()
    ekf_mean = 1
    ekf_cov = 1
    measure_x = kalman.make_data()
    predict_list = []
    for i in range(100):
        mean, cov = kalman.Extended_kalman(ekf_mean,ekf_cov,i,measure_x)
        ekf_mean = mean
        print(mean)
        ekf_cov = cov
        predict_list.append(mean)
    x = range(100)
    print(predict_list)
    plt.plot(measure_x,'r-',predict_list)
    plt.title('EKF')
    plt.legend(['real','predict'])
    plt.show()

