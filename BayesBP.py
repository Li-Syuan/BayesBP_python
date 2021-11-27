# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:58:10 2020

@author: LiSyuan Hong
"""

#from numba import jit
import numpy as np
import scipy
from scipy.special import comb
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from joblib import Parallel, delayed
import time


class Simulated_Data:
    def __init__(self, ages, years):
        self.ages = ages
        self.years = years
        x = self.map_to_01(ages)
        y = self.map_to_01(years)
        self.x, self.y = np.meshgrid(x, y)

    def map_to_01(self, x):
        mn = np.min(x)
        mx = np.max(x)
        r = (x - mn) / (mx - mn)
        return r

    def FT(self, x, y):
        value = 0.00148 * np.sin(x * (y+.02) * np.pi * 0.5) + 0.00002
        return value

    def M(self, x, y):
        r0 = 152040
        r1 = -285270 * x
        r2 = 110410 * y
        r3 = 173990 * np.power(x,2)
        r4 = -49950 * x * y
        r5 = -33630 * np.power(y,2) 
        r6 = -19530 * np.power(x,3)
        r7 = -110330 * np.power(x,2) * y
        r8 = 88840 * x * np.power(y,2) 
        r9 = -7990 * np.power(y,3)     
        value = r0 + r1 + r2+ r3 + r4 + r5 + r6 + r7 + r8 + r9                                                                            
        return value

    def Generate(self):
        incidence = self.FT(self.x, self.y)
        population = self.M(self.x, self.y)
        disease = incidence * population
        return disease, population


class BayesBP():
    def __init__(self, prior, ages, years, disease, population):
        # input data
        self.n0, self.alpha, self.LL = prior
        self.ages = ages
        self.years = years
        self.disease = disease
        self.population = population
        # incidence rate
        self.rate = disease / population
        # linear tranfomation
        x = self.__map_to_01(ages)
        y = self.__map_to_01(years)
        self.__lx = len(x)
        self.__ly = len(y)
        self.__lxy = len(x) * len(y)
        x, y = np.meshgrid(x, y)
        self.__x1 = np.reshape(x, self.__lx * self.__ly)
        self.__y1 = np.reshape(y, self.__lx * self.__ly)
        # build basis
        self.__listbasis = self.__BPbasis()

    def set_parameter(self,
                      itr=200000,
                      nchain=5,
                      RJc=.35,
                      stocoef=100,
                      nn=2,
                      seed=123,
                      num_cores=5,
                      double=4):
        self.__itr = itr  # length of chains
        self.__nchain = nchain  # number of chains
        self.__RJc = RJc  # RJ constant
        self.__stocoef = stocoef  # Stored coefficients
        self.__nn = nn  # Possion lower bound
        self.__seed = seed  # Random seed
        self.__num_cores = num_cores  # Number of cores
        self.__double = double  # Double times
        res = {'Iterations': itr,
               'Number of chain': nchain,
               'Random jump parameter': RJc,
               'Stored coeffient': stocoef,
               'Possion lower bound': nn,
               'Random seed': seed,
               'Number of cores': num_cores,
               'Double times': double}
        return res

    def __map_to_01(self, x):
        mn = np.min(x)
        mx = np.max(x)
        r = (x - mn) / (mx - mn)
        return r

    # binomial function
    def __Bin(self, n, i, x):
        r = comb(n, i) * x**i * (1 - x)**(n - i)
        return r

    # Bernstein basis
    def __BPbasis(self):
        listbasis = [0]

        def bp2d(i, j):
            a = self.__Bin(n, i, self.__x1[k])
            b = self.__Bin(n, j, self.__y1[k])
            return a * b
        for n in range(1, self.n0 + 1):
            g = np.zeros((self.__lxy, n + 1, n + 1))
            i = range(n + 1)
            j = range(n + 1)
            i, j = np.meshgrid(i, j)
            for k in range(self.__lxy):
                g[k, i, j] = bp2d(i, j)
            listbasis.append(g)
        return listbasis

    # Bernstein polynomial
    def __BP(self, a):
        n = a.shape[0]
        aa = np.resize(a, (self.__lxy, n, n))
        aa = (aa * self.__listbasis[n - 1]).flatten().reshape(self.__lxy, n**2)
        aa = aa.sum(axis=1)
        return np.mean(aa), np.max(aa)

    # outer Bernstein polynomial
    def __BPFhat(self, a):
        n = a.shape[0]
        aa = np.resize(a, (self.__lxy, n, n))
        aa = (aa * self.__listbasis[n - 1]).flatten().reshape(self.__lxy, n**2)
        aa = aa.sum(axis=1)
        aa = aa.reshape(self.__ly, self.__lx)
        return aa

    # specific P-value
    def __Pvalue(self, a):
        M = self.__BPFhat(a)
        M1 = np.random.poisson(M * self.population) / self.population
        EZR1 = np.max(np.abs(M1 - M))
        EZR2 = np.abs(np.mean(M1) - np.mean(M))
        ZR1 = np.max(np.abs(M - self.rate))
        ZR2 = np.abs(np.mean(M) - np.mean(self.rate))
        PMax = EZR1 > ZR1
        PMean = EZR2 > ZR2
        return PMean, PMax

    # log likihood function
    def __logLF(self, a):
        n = a.shape[0]
        aa = np.resize(a, (self.__lxy, n, n))
        aa = (aa * self.__listbasis[n - 1]).flatten().reshape(self.__lxy, n**2)
        aa = aa.sum(axis=1)
        aa = aa.reshape(self.__ly, self.__lx)
        MM = aa * self.population
        pop = scipy.special.loggamma(self.population + 1)
        return np.sum(self.disease *np.log(MM) -MM - pop)

    # set prior infomation
    def initialize(self):
        self.__M2 = np.max(self.rate) * self.LL
        self.__M1 = 0
        self.__l1 = np.max(self.rate[0, :]) * self.LL
        self.__l2 = np.max(self.rate[self.rate.shape[0] - 1, :]) * self.LL
        self.__l3 = np.max(self.rate[:, 0]) * self.LL
        self.__l4 = np.max(self.rate[:, self.rate.shape[1] - 1]) * self.LL
        self.__ld = np.array([self.__l1, self.__l2, self.__l3, self.__l4])
        # set possion p
        p = np.sum([self.__pois(x, self.alpha)
                    for x in range(self.__nn + 1)])
        self.__p1 = [self.__pois(x, self.alpha)
                     for x in range(self.__nn + 1, self.n0)]
        p2 = 1 - sum(self.__p1) - p
        self.__p1.insert(0, p)
        self.__p1.append(p2)
        
        np.random.seed(self.__seed)
        n1 = np.random.choice(np.arange(self.__nn, self.n0 + 1),
                              size = self.__nchain,
                              replace = True, p  =self.__p1)
        for i in range(self.__nn + 1):
            self.__p1.insert(0, 0)
        self.__p1.append(0)
        n1 = pd.Series(n1)
        self.__initialvalue = n1.apply(self.__initial_parameter)

    # define chosen poisson probility
    def __pois(self, x, u):
        return np.exp(-u) * u**x / scipy.special.gamma(x + 1)

    # set initial parameter
    def __initial_parameter(self, n1):
        a = n1 + 1
        b = n1 - 1 
        h1 = np.random.uniform(self.__M1, self.__l1, a)
        h2 = np.random.uniform(self.__M1, self.__l2, a)
        h3 = np.random.uniform(self.__M1, self.__l3, b)
        h4 = np.random.uniform(self.__M1, self.__l4, b)
        h5 = np.random.uniform(self.__M1, self.__M2, b**2).reshape(b, b)
        h6 = np.c_[h3, h5, h4]
        h7 = np.vstack((h1, h6, h2))
        return h7

    # Gelman Rubin statistic
    # check chains convergence
    def Rhat(self, M, burnin=0.5):
        m = M.shape[0]
        x = M[:, np.arange(int(np.round(burnin * M.shape[1])), M.shape[1])]
        n = x.shape[1]
        phibar = np.mean(x)
        phi = np.mean(x, axis=1)
        B = n / (m - 1) * np.sum((phi - phibar)**2)
        W = np.mean(np.var(x, axis=1))
        postvar = (n - 1) / n * W + B / n
        return np.sqrt(postvar / W)

    def __norm(self):
        tmp = self.rate - self.result['Bayes estimation']
        self.norm = {'L1-norm' : np.abs(tmp).mean() * 10**5,
                     'L2-norm' : np.linalg.norm(tmp, 2) * 10**5,
                     'Sup-norm' : np.abs(tmp).max() * 10**5,
                     'RMSE' : np.sqrt(np.power(tmp, 2).mean()) * 10**5
                     }
    def __MCMC(self, initialvalue, seed, kk=0.5):
        np.random.seed(seed)
        g = self.__itr * kk
        store = []
        storeF = np.zeros((self.__ly, self.__lx))
        chain = []
        maxchain = []
        Pmean = []
        Pmax = []
        R = np.array(initialvalue)
        n1 = R.shape[0]

        rep = 0
        if kk == 0.5:
            rep = 0
        else:
            rep = kk * self.__itr

        for i in np.arange(rep, 2 * kk * self.__itr):
            PP = [min(1,self.__p1[n1 - 1] / self.__p1[n1]),
                  min(1,self.__p1[n1 + 1] / self.__p1[n1])]
            RJP = [self.__RJc * PP[0],
                   self.__RJc * PP[1]]
            prob = [RJP[0], 1 - np.sum(RJP), RJP[1]]

            if prob[0] == 0:
                H = np.random.choice([1, 2], 1, p=prob[1:3])
            elif prob[2] == 0:
                H = np.random.choice([0, 1], 1, p=prob[0:2])
            else:
                H = np.random.choice([0, 1, 2], 1, p=prob)
                
            if H == 0:
                n2 = n1 - 1
                delete = np.random.choice(n1, 2)
                S = np.delete(R, delete[0], axis=0)
                S = np.delete(S, delete[1], axis=1)
                rio = self.__logLF(S) - self.__logLF(R) + np.log(n2) - sum(
                    np.log(self.__ld - self.__M1)) - np.log(self.__M2 - self.__M1) * (2 * n2 - 3)

            elif H == 2:
                n2 = n1 + 1
                add = np.random.choice(n1-1, 2)
                m1 = R[:(add[0]+1),:]
                m2 = R[add[0]:n1, :]
                m = np.r_[m1, m2]
                m1 = m[:, :(add[1] + 1)]
                m2 = m[:,add[1]:n1]
                S = np.c_[m1, m2]
                ww = np.r_[
                    np.random.uniform(self.__M1, self.__l3, 1),
                    np.random.uniform(self.__M1, self.__M2, n2 - 2),
                    np.random.uniform(self.__M1, self.__l4, 1)]
                vv = np.r_[
                    np.random.uniform(self.__M1, self.__l1, 1),
                    np.random.uniform(self.__M1, self.__M2, n2 - 2),
                    np.random.uniform(self.__M1, self.__l2,1)]
                S[add[0] + 1, :] = ww
                S[:, add[1] + 1] = vv
                S = np.asarray(S)
                rio = self.__logLF(S) - self.__logLF(R) - np.log(n2) + np.sum(
                    np.log(self.__ld - self.__M1)) + np.log(self.__M2 - self.__M1) * (2 * n2 - 1)

            else:
                n2 = n1
                S = R + 0
                if n2 == 1:
                    stay = np.random.choice(4)
                else:
                    stay = np.random.choice(9)
                if stay == 0:
                    S[0, 0] = np.random.uniform(self.__M1, self.__l1, 1)
                elif stay == 1:
                    S[0, -1] = np.random.uniform(self.__M1, self.__l1, 1)
                elif stay == 2:
                    S[-1, 0] = np.random.uniform(self.__M1, self.__l2, 1)
                elif stay == 3:
                    S[-1, -1] = np.random.uniform(self.__M1, self.__l2, 1)
                elif stay == 4:
                    S[0,1:-1] = np.random.uniform(self.__M1,self.__l1,n2 -2 )
                elif stay == 5:
                    S[-1,1:-1] = np.random.uniform(self.__M1,self.__l2,n2 -2)
                elif stay == 6:
                    S[1:-1,0] = np.random.uniform(self.__M1,self.__l3,n2-2)
                elif stay == 7:
                    S[1:-1, -1] = np.random.uniform(self.__M1, self.__l4, n2 -2)
                else:
                    ch = np.random.choice(n2-1, 1)+1
                    S[ch, 1:-1] = np.random.uniform(self.__M1,self.__M2,n2 -2)
                rio = self.__logLF(S) - self.__logLF(R)

            if rio > 0:
                nnext = n2
                xnext = S
            else:
                if np.log(np.random.random_sample()) < rio:
                    nnext = n2
                    xnext = S
                else:
                    nnext = n1
                    xnext = R
            R = xnext
            n1 = nnext

            if i > g and(i % g % self.__stocoef == 1):
                chain += [self.__BP(xnext)[0]]
                maxchain += [self.__BP(xnext)[1]]
                temp = self.__Pvalue(xnext)
                Pmean.append(temp[0])
                Pmax.append(temp[1])
                store.append(xnext)
                storeF += self.__BPFhat(xnext)
            del xnext, nnext
        estF = storeF / len(chain)

        return chain, maxchain, estF, store, Pmean, Pmax

    def RJMHalgorithm(self):
        Start = time.time()
        # loky threading multiprocessing
        Rhat = 10
        kk = .5
        doubletime = 0
        print('#', '-' * 30, '#')
        np.random.seed(self.__seed)
        seed = np.random.choice(range(1000), self.__nchain)
        while Rhat > 1.1:
            results = Parallel(n_jobs=self.__num_cores)(
                delayed(self.__MCMC)(
                    initialvalue=self.__initialvalue[i],
                    seed=seed[i],
                    kk=kk)
                for i in range(self.__nchain))
            End = time.time()
            print("# Cost time: %.3f mins " % float((End - Start) / 60))
            Fhat = 0
            self.store_coefficient = []
            chain, maxchain, Pmean, Pmax = [], [], [], []
            for i in range(self.__nchain):
                chain.append(results[i][0])
                maxchain.append(results[i][1])
                Fhat += results[i][2]
                self.store_coefficient.append(results[i][3])
                Pmean.append(results[i][4])
                Pmax.append(results[i][5])

            chain = np.array(chain)
            maxchain = np.array(maxchain)
            Rhat = self.Rhat(chain, burnin=0)

            self.result = {'Posterior mean': chain,
                           'Posterior max': maxchain,
                           'Bayes estimation': Fhat / self.__nchain,
                           'Pvalue mean': np.mean(Pmean),
                           'Pvalue max': np.mean(Pmax)}

            self.__norm()
            print('# L1norm: %.5f' % self.norm['L1-norm'])
            print('# L2norm: %.5f' % self.norm['L2-norm'])
            print('# supnorm: %.5f' % self.norm['Sup-norm'])
            print('# RMSE: %.5f' % self.norm['RMSE'])
            tmp = self.result['Bayes estimation']
            print('# Bayes estimation mean: %.5f' % (np.mean(tmp) * 10**5))
            print('# Bayes estimation max: %.5f' % (np.max(tmp) * 10**5))
            print('# Iterations:', int(self.__itr * kk * 2))
            print('# Pvalue mean:', np.mean(Pmean))
            print('# Pvalue max:', np.mean(Pmax))
            print('# Rhat of posterior mean: %.5f' % Rhat)
            print('# Rhat of posterior max: %.5f' %
                self.Rhat(maxchain,burnin=0))
            if Rhat < 1.1:
                print('# Markov chains is convergence.')
            else:
                print('# Markov chains is  not convergence.')
                print('# Doubling iterations')
                print('#', '-' * 30, '#')
                kk *= 2
                doubletime += 1
            if doubletime == self.__double:
                del chain, maxchain, Pmean, Pmax, Fhat, tmp
                break
            else:
                self.__initialvalue = np.array(
                    [self.store_coefficient[i][-1].tolist() for i in range(5)])
        print('#', '-' * 30, '#')
        self.result

    def traceplot_kde(self):
        df = pd.DataFrame(self.result['Posterior max'].transpose() * 10**5)
        df.plot(title='Posterior max trace plot', kind='line')
        df.plot(title='Posterior max density plot', kind='kde')
        df = pd.DataFrame(self.result['Posterior mean'].transpose() * 10**5)
        df.plot(title='Posterior mean trace plot', kind='line')
        df.plot(title='Posterior mean density plot', kind='kde')

    def credible_interval(self,alpha=.05):
        compute=[]
        for i in range(self.__nchain):
            for coef in self.store_coefficient[i]:
                compute.append(
                    self.__BPFhat(coef).tolist())
        compute=np.asarray(compute)
        CI=np.zeros((2,self.__ly,self.__lx))
        for i in range(self.__ly):
            for j in range(self.__lx):
                CI[:,i,j]=[
                    np.quantile(compute[:,i,j],alpha/2),
                    np.quantile(compute[:,i,j],1-alpha/2)]
        self.CI = CI 
        return self.CI

    
if __name__ == '__main__':
    # possion parameter
    # 1. Upper bound
    # 2. possion mean
    # 3. mutiple parameter
    prior = [10, 5, 2]

    ages = np.arange(35, 85 + 1)
    years = np.arange(1988, 2007 + 1)
    disease, population = Simulated_Data(ages, years).Generate()

    BP = BayesBP(prior, ages, years, disease, population)
    BP.set_parameter(itr=200000,
                     nchain=5,
                     RJc=.35,
                     stocoef=100,
                     nn=3,
                     seed=123678,
                     num_cores=5,
                     double=4)
    # set initial Bernstein polynomial parameter
    BP.initialize()
    # run algorithm
    BP.RJMHalgorithm()
    # get result
    BP.result
    BP.norm
    #compute credible interval
    BP.credible_interval()
    BP.CI
    # trace plot and density plot
    BP.traceplot_kde()
    #BP._BayesBP__initialvalue[0]
    #BP._BayesBP__MCMC(BP._BayesBP__initialvalue[0],seed=1)









