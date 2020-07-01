import numpy as np
import matplotlib.pyplot as plt
import math

class ekf_radar:
    def __init__(self):
        self.pre_pose = 0
        self.T = 0.05
        self.R = 10


    def Hjacobian(self,x):
        x1 = x[0]
        x3 = x[2]
        out = []
        x1 = (x1)/np.sqrt((x1 ** 2)+(x3 ** 2))
        x3 = (x3)/np.sqrt((x1 ** 2)+(x3 ** 2))
        out.append((x1,0,x3))
        return out

    def Hx(self,x):
        x1 = x[0]
        x3 = x[2]
        result = np.sqrt((x1 ** 2) + (x3 ** 2))
        return result

    def main(self, z, pre_mean, pre_cov):

        A = np.array([[1,self.T,0],[0,1,0],[0,0,1]])
        Q = np.array([[0,0,0],[0,0.001,0],[0,0,0.001]])
        H = self.Hjacobian(pre_mean)
        AT = np.transpose(A)
        HT = np.transpose(H)
        mean_hat = np.dot(A,pre_mean,)
        cov_hat = np.dot(np.dot(A,pre_cov), AT,) + Q
        print('Ax meanhat', mean_hat)
        print('cov_hat:',cov_hat)
        K = np.dot(np.dot(cov_hat, HT), np.linalg.pinv(np.dot(np.dot(H, cov_hat), HT) + self.R))

        Hx = self.Hx(mean_hat)
        # print('K:',K)
        x = mean_hat + K*(z - Hx)
        x = mean_hat + np.dot(K,(z-Hx))
        # print('x',x.shape)
        # print('x',x)
        cov = cov_hat - np.dot(np.dot(K,H),cov_hat)
        self.pre_mean = x
        self.pre_cov = cov
        # print('K:',K)
        print('result:',x)
        return x,cov

    def radardata(self):

        vel = 100 + 5*np.random.randn()
        alt = 1000 + 10*np.random.randn(1)

        pose = self.pre_pose + vel*self.T

        v = 0 + pose*0.05 * np.random.randn(1)
        r = np.sqrt((pose ** 2) + (alt ** 2)) + v
        self.pre_pose= pose
        return r, pose



ekf = ekf_radar()
pose_list = []
vel_list = []
alt_list = []
real_list = []
pre_mean = [0, 90, 1100]
pre_cov = 10 * np.eye(3)

for i in range(0,20):

    # r, real_pose = ekf.radardata()
    # real_mean, real_cov = ekf.main(r,pre_mean,pre_cov)
    # pose_list.append(real_mean[0])
    # vel_list.append(real_mean[1])
    # alt_list.append(real_mean[2])
    # real_list.append(real_pose)
    print(hello)
#
    pre_mean = real_mean
    pre_cov = real_cov

real_list = np.array(real_list)
pose_list = np.array(pose_list)
x = range(20)
plt.plot(x,real_list,x,pose_list)
plt.show()