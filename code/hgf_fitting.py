# -*- coding: utf-8 -*-

"""
Authors: Zheng Wang
input: emp sc emp eeg stimulus and behavior data
outputs: sim eeg and behavior (fitted)
"""

import numpy as np

class hgf_fitting:

    def __init__(self) -> None:
        pass

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_dxx(self, x):
        return (np.exp(-2*x)-np.exp(-x))/(1+np.exp(-x))**3

    def sigmoid_dx(self, x):
        return np.exp(-x)/(1+np.exp(-x))**2


    # relu function to make sure variable positive
    def relu(self, x):
        if x <0.01:
          return 0.01
        else:
          return x
    # function to make sure hess is not too small
    def relu_abs(self, x):
        if np.abs(x) <0.1:
          return np.sign(x)*0.1
        else:
          return x

    def hess_grad_mid(self, x2, mu2, var2, k, w, x3):
        # F = 0.5*ln(varhat2)+0.5*s2^2
        varhat2 = var2+np.exp(k*x3+w)
        s2 = (x2-mu2)/np.sqrt(varhat2)

        # gradient to sighat and s from cost(free energy)
        dFdvarhat2 = 0.5/varhat2
        dFds2 = s2


        # gradient to states and mean variance of states from sigmahat and s3
        ds2dx2 = 1/np.sqrt(varhat2)
        ds2dmu2 = - 1/np.sqrt(varhat2)
        dvarhat2dx3 = k*np.exp(k*x3+w)
        ds2dx3 = -0.5*(x2-mu2)*np.power(varhat2, -1.5)*dvarhat2dx3



        dvarhat2dvar2 = 1
        ds2dvar2 = -0.5*(x2-mu2)*np.power(varhat2, -1.5)*dvarhat2dvar2

        # gradient to paramters(k, w) from sigmahat and s3 (not necissary)
        dvarhat2dk = x3*np.exp(k*x3+w)
        dvarhat2dw = np.exp(k*x3+w)
        ds2dk = -0.5*(x2-mu2)*np.power(varhat2, -1.5)*dvarhat2dk
        ds2dw = -0.5*(x2-mu2)*np.power(varhat2, -1.5)*dvarhat2dw

        Fdx2 = dFds2*ds2dx2 # s2/np.sqrt(varhat2)
        Fdx3 = dFdvarhat2*dvarhat2dx3 +dFds2*ds2dx3 
              #s2*(-0.5*(x2-mu2)*np.power(varhat2, -1.5)*dvarhat2dx3)
              # +0.5/varhat2*k*np.exp(k*x3+w)

        Fdmu2 = dFds2*ds2dmu2 #-s2/np.sqrt(varhat2)
        

        Fdvar2 = dFds2*ds2dvar2 + dFdvarhat2*dvarhat2dvar2
                # s2*(-0.5*(x2-mu2)*np.power(varhat2, -1.5)*dvarhat2dvar2)
                # 0.5/varhat2
        
        Fdk =dFdvarhat2*dvarhat2dk + dFds2*ds2dk
            #s2*(-0.5*(x2-mu2)*np.power(varhat2, -1.5)*dvarhat2dk)
              # +0.5/varhat2*x3*np.exp(k*x3+w)
        Fdw =dFdvarhat2*dvarhat2dw + dFds2*ds2dw
            #s2*(-0.5*(x2-mu2)*np.power(varhat2, -1.5)*dvarhat2dw)
              # +0.5/varhat2*np.exp(k*x3+w)

        Fddx2 = ds2dx2*ds2dx2
        Fddx3 = -0.5/varhat2**2*dvarhat2dx3**2 + 0.5/varhat2*k*dvarhat2dx3 \
                +ds2dx3**2 -1.5*s2*ds2dx3*dvarhat2dx3/varhat2\
                +k*s2*ds2dx3
        
        Fddmu2 = ds2dmu2*ds2dmu2
        
        Fddvar2 = ds2dvar2*ds2dvar2 -1.5*s2/varhat2*ds2dvar2*dvarhat2dvar2\
                  -0.5/varhat2**2*dvarhat2dvar2
        
        Fddw = -0.5/varhat2**2*dvarhat2dw**2 +0.5/varhat2*dvarhat2dw\
              +ds2dw*ds2dw - 1.5*s2*ds2dw/varhat2*dvarhat2dw\
              +s2*ds2dw
        
        Fddk = -0.5/varhat2**2*dvarhat2dk**2 +0.5/varhat2*x3*dvarhat2dk\
              +ds2dk*ds2dk - 1.5*s2*ds2dk/varhat2*dvarhat2dk\
              +s2*ds2dk*x3

        grad= np.array([Fdx2, Fdmu2, Fdvar2, Fdk, Fdw, Fdx3])
        hess = np.array([Fddx2, Fddmu2, Fddvar2, Fddk, Fddw, Fddx3])

        return grad, hess

    def hess_grad_mid_res(self, x2, mu2, var2, k, w, p, x3, u, v):
        # F = 0.5*ln(varhat2)+0.5*s2^2
        varhat2 = var2+np.exp(k*x3+w)+0.0001
        s2 = (x2-mu2 - p*u -v)/np.sqrt(varhat2)

        # gradient to sighat and s from cost(free energy)
        dFdvarhat2 = 0.5/varhat2
        dFds2 = s2


        # gradient to states and mean variance of states from sigmahat and s3
        ds2dx2 = 1/np.sqrt(varhat2)
        ds2dmu2 = - 1/np.sqrt(varhat2)
        dvarhat2dx3 = k*np.exp(k*x3+w)
        ds2dx3 = -0.5*(x2-mu2-p*u-v)*np.power(varhat2, -1.5)*dvarhat2dx3



        dvarhat2dvar2 = 1
        ds2dvar2 = -0.5*(x2-mu2-p*u-v)*np.power(varhat2, -1.5)*dvarhat2dvar2

        # gradient to paramters(k, w) from sigmahat and s3 (not necissary)
        dvarhat2dk = x3*np.exp(k*x3+w)
        dvarhat2dw = np.exp(k*x3+w)
        
        ds2dk = -0.5*(x2-mu2-p*u-v)*np.power(varhat2, -1.5)*dvarhat2dk
        ds2dw = -0.5*(x2-mu2-p*u-v)*np.power(varhat2, -1.5)*dvarhat2dw
        ds2dp = - u/np.sqrt(varhat2)

        Fdx2 = dFds2*ds2dx2 # s2/np.sqrt(varhat2)
        Fdx3 = dFdvarhat2*dvarhat2dx3 +dFds2*ds2dx3 
              #s2*(-0.5*(x2-mu2)*np.power(varhat2, -1.5)*dvarhat2dx3)
              # +0.5/varhat2*k*np.exp(k*x3+w)

        Fdmu2 = dFds2*ds2dmu2 #-s2/np.sqrt(varhat2)
        

        Fdvar2 = dFds2*ds2dvar2 + dFdvarhat2*dvarhat2dvar2
                # s2*(-0.5*(x2-mu2)*np.power(varhat2, -1.5)*dvarhat2dvar2)
                # 0.5/varhat2
        
        Fdk =dFdvarhat2*dvarhat2dk + dFds2*ds2dk
            #s2*(-0.5*(x2-mu2)*np.power(varhat2, -1.5)*dvarhat2dk)
              # +0.5/varhat2*x3*np.exp(k*x3+w)
        Fdw =dFdvarhat2*dvarhat2dw + dFds2*ds2dw
            #s2*(-0.5*(x2-mu2)*np.power(varhat2, -1.5)*dvarhat2dw)
              # +0.5/varhat2*np.exp(k*x3+w)

        Fdp =  dFds2*ds2dp

        Fddx2 = ds2dx2*ds2dx2
        Fddx3 = -0.5/varhat2**2*dvarhat2dx3**2 + 0.5/varhat2*k*dvarhat2dx3 \
                +ds2dx3**2 -1.5*s2*ds2dx3*dvarhat2dx3/varhat2\
                +k*s2*ds2dx3
        
        Fddmu2 = ds2dmu2*ds2dmu2
        
        Fddvar2 = ds2dvar2*ds2dvar2 -1.5*s2/varhat2*ds2dvar2*dvarhat2dvar2\
                  -0.5/varhat2**2*dvarhat2dvar2
        
        Fddw = -0.5/varhat2**2*dvarhat2dw**2 +0.5/varhat2*dvarhat2dw\
              +ds2dw*ds2dw - 1.5*s2*ds2dw/varhat2*dvarhat2dw\
              +s2*ds2dw
        
        Fddk = -0.5/varhat2**2*dvarhat2dk**2 +0.5/varhat2*x3*dvarhat2dk\
              +ds2dk*ds2dk - 1.5*s2*ds2dk/varhat2*dvarhat2dk\
              +s2*ds2dk*x3
        Fddp = ds2dp**2

        grad= np.array([Fdx2, Fdmu2, Fdvar2, Fdk, Fdw, Fdp, Fdx3])
        hess = np.array([Fddx2, Fddmu2, Fddvar2, Fddk, Fddw, Fddp, Fddx3])

        return grad, hess
    def hess_grad_top(self, x, mu, var, theta):
        # F = 0.5*ln(varhat)+0.5*s^2
        varhat = var+theta
        s = (x-mu)/np.sqrt(varhat)

        # gradient to sighat and s from cost(free energy)
        dFdvarhat = 0.5/varhat
        dFds = s


        # gradient to states and mean variance of states from sigmahat and s3
        dsdx = 1/np.sqrt(varhat)
        dsdmu = - 1/np.sqrt(varhat)

        dvarhatdvar = 1
        dsdvar = -0.5*(x-mu)*np.power(varhat, -1.5)*dvarhatdvar

        # gradient to paramters theta from sigmahat and s3 (not necissary)
        dvarhatdtheta = 1
        dsdtheta = -0.5*(x-mu)*np.power(varhat, -1.5)*dvarhatdtheta

        Fdx = dFds*dsdx # s/np.sqrt(varhat)
        Fdmu = dFds*dsdmu # -s/np.sqrt(varhat)
        

        Fdvar = dFds*dsdvar + dFdvarhat*dvarhatdvar #(-s*0.5*(x-mu)*np.power(varhat, -1.5) +0.5/varhat)
        
        Fdtheta =dFdvarhat*dvarhatdtheta + dFds*dsdtheta #(-s*0.5*(x-mu)*np.power(varhat, -1.5) +0.5/varhat)
        
        Fddx = dsdx*dsdx
        
        Fddmu = dsdmu*dsdmu
        
        Fddvar = dsdvar*dsdvar -1.5*s/varhat*dsdvar*dvarhatdvar -0.5/varhat**2*dvarhatdvar
        
        Fddtheta = dsdtheta*dsdtheta -1.5*s/varhat*dsdtheta*dvarhatdtheta -0.5/varhat**2*dvarhatdtheta
        grad= np.array([Fdx, Fdmu, Fdvar, Fdtheta])
        hess = np.array([Fddx, Fddmu, Fddvar, Fddtheta])

        return grad, hess   

    def hess_grad_response(self, x, mu, var, mu2, var2, p1, p2, theta, u):
        # F = 0.5*ln(varhat)+0.5*s^2
        varhat = var+theta + p1*var2
        s = (x-mu -p1*mu2- p2*(u-0.0))/np.sqrt(varhat)

        # gradient to sighat and s from cost(free energy)
        dFdvarhat = 0.5/varhat
        dFds = s


        # gradient to states and mean variance of states from sigmahat and s3
        dsdx = 1/np.sqrt(varhat)
        dsdmu = - 1/np.sqrt(varhat)
        dsdmu2 = -p1/np.sqrt(varhat)

        dvarhatdvar2 = p1
        dsdvar2 = -0.5*(x-mu+p1*mu2+ p2*(u-0.0))*np.power(varhat, -1.5)*dvarhatdvar2

        dvarhatdvar = 1
        dsdvar = -0.5*(x-mu-p1*mu2 - p2*(u-0.0))*np.power(varhat, -1.5)*dvarhatdvar

        # gradient to paramters theta from sigmahat and s (not necissary)
        dvarhatdtheta = 1
        dsdtheta = -0.5*(x-mu-p1*mu2- p2*(u-0.0))*np.power(varhat, -1.5)*dvarhatdtheta

        dvarhatdp1 = var2
        dsdp1 = -mu2/np.sqrt(varhat) -0.5*(x-mu-p1*mu2- p2*(u-0.0))*np.power(varhat, -1.5)*dvarhatdp1

        dsdp2 =  -(u-0.0)/np.sqrt(varhat)


        Fdx = dFds*dsdx # s/np.sqrt(varhat)
        Fdmu = dFds*dsdmu # -s/np.sqrt(varhat)
        Fdvar = dFdvarhat*dvarhatdvar + dFds*dsdvar
            #s*(-0.5*(x-mu-p1*mu2- p2*(u-0.0))*np.power(varhat, -1.5)*dvarhatdvar)
            #+0.5/varhat
        Fdmu2 = dFds*dsdmu2# -p1*s/np.sqrt(varhat)
        Fdvar2 = dFdvarhat*dvarhatdvar2 + dFds*dsdvar2
            #s*(-0.5*(x-mu-p1*mu2- p2*(u-0.5))*np.power(varhat, -1.5)*dvarhatdvar2)
            # +0.5/varhat*p1
        Fdtheta = dFdvarhat*dvarhatdtheta + dFds*dsdtheta
            # s*(-0.5*(x-mu-p1*mu2- p2*(u-0.5))*np.power(varhat, -1.5)*dvarhatdtheta)
            # +0.5/varhat
          
        Fdp1 = dFdvarhat*dvarhatdp1 + dFds*dsdp1
          # s*(-mu2/np.sqrt(varhat) -0.5*(x-mu-p1*mu2- p2*(u-0.5))*np.power(varhat, -1.5)*dvarhatdp1)
          # +0.5/varhat*var2
        Fdp2 = dFds*dsdp2
          # -s*(u-0.5)/np.sqrt(varhat) 

        Fddx = dsdx*dsdx
        Fddmu = dsdmu*dsdmu
        Fddvar = -0.5/varhat**2*dvarhatdvar**2 +dsdvar*dsdvar -1.5*dFds*np.power(varhat, -1)*dsdvar*dvarhatdvar
        Fddmu2 = dsdmu2*dsdmu2
        Fddvar2 = -0.5/varhat**2*dvarhatdvar2**2 + dsdvar2*dsdvar2 -1.5*dFds*np.power(varhat, -1)*dsdvar2*dvarhatdvar2

        Fddtheta = -0.5/varhat**2*dvarhatdtheta**2 + dsdtheta*dsdtheta -1.5*dFds*np.power(varhat, -1)*dsdtheta*dvarhatdtheta
        Fddp1 = dsdp1*dsdp1 +1.5*s*dvarhatdp1*0.5*(x-mu-p1*mu2- p2*(u-0.0))*np.power(varhat, -2.5)*dvarhatdp1**2\
              + 1.0*s*mu2*np.power(varhat, -1.5)*dvarhatdp1\
              -0.5/varhat**2*dvarhatdp1**2
        Fddp2 = dsdp2*dsdp2

        grad= np.array([Fdx, Fdmu, Fdvar, Fdtheta, Fdp1, Fdp2, Fdmu2, Fdvar2])
        hess = np.array([Fddx, Fddmu, Fddvar, Fddtheta, Fddp1, Fddp2, Fddmu2, Fddvar2])

        return grad, hess 

    def hess_grad_response_new(self, x, mu, var, mu2, p1, p2, theta, u):
        # F = 0.5*ln(varhat)+0.5*s^2
        varhat = var+theta 
        s = (x-mu +p1*mu2+ p2*(u-0.5))/np.sqrt(varhat)

        # gradient to sighat and s from cost(free energy)
        dFdvarhat = 0.5/varhat
        dFds = s

        
        # gradient to states and mean variance of states from sigmahat and s3
        dsdx = 1/np.sqrt(varhat)
        dsdmu = - 1/np.sqrt(varhat)
        dsdmu2 = p1/np.sqrt(varhat)

        
        dvarhatdvar = 1
        dsdvar = -0.5*(x-mu+p1*mu2+ p2*(u-0.5))*np.power(varhat, -1.5)*dvarhatdvar

        # gradient to paramters theta from sigmahat and s (not necissary)
        dvarhatdtheta = 1
        dsdtheta = -0.5*(x-mu+p1*mu2+ p2*(u-0.5))*np.power(varhat, -1.5)*dvarhatdtheta

        dvarhatdp1 = 0
        dsdp1 = mu2/np.sqrt(varhat) -0.5*(x-mu+p1*mu2+ p2*(u-0.5))*np.power(varhat, -1.5)*dvarhatdp1

        dsdp2 =  (u-0.5)/np.sqrt(varhat)


        Fdx = dFds*dsdx # s/np.sqrt(varhat)
        Fdmu = dFds*dsdmu # -s/np.sqrt(varhat)
        Fdvar = dFdvarhat*dvarhatdvar + dFds*dsdvar
            #s*(-0.5*(x-mu+p1*mu2+ p2*(u-0.5))*np.power(varhat, -1.5)*dvarhatdvar)
            #+0.5/varhat
        Fdmu2 = dFds*dsdmu2# p1*s/np.sqrt(varhat)
        
        Fdtheta = dFdvarhat*dvarhatdtheta + dFds*dsdtheta
            # s*(-0.5*(x-mu+p1*mu2+ p2*(u-0.5))*np.power(varhat, -1.5)*dvarhatdtheta)
            # +0.5/varhat
          
        Fdp1 = dFdvarhat*dvarhatdp1 + dFds*dsdp1
          # s*(mu2/np.sqrt(varhat) -0.5*(x-mu+p1*mu2+ p2*(u-0.5))*np.power(varhat, -1.5)*dvarhatdp1)
          # +0.5/varhat*var2
        Fdp2 = dFds*dsdp2
          # s*(u-0.5)/np.sqrt(varhat) 

        Fddx = dsdx*dsdx
        Fddmu = dsdmu*dsdmu
        Fddvar = -0.5/varhat**2*dvarhatdvar**2 +dsdvar*dsdvar -1.5*dFds*np.power(varhat, -1)*dsdvar*dvarhatdvar
        Fddmu2 = dsdmu2*dsdmu2
        

        Fddtheta = -0.5/varhat**2*dvarhatdtheta**2 + dsdtheta*dsdtheta -1.5*dFds*np.power(varhat, -1)*dsdtheta*dvarhatdtheta
        Fddp1 = dsdp1*dsdp1 -1.5*s/varhat*dsdp1*dvarhatdp1- 0.5*s*mu2*np.power(varhat, -1.5)*dvarhatdp1\
              -0.5/varhat**2*dvarhatdp1**2
        Fddp2 = dsdp2*dsdp2

        grad= np.array([Fdx, Fdmu, Fdvar, Fdtheta, Fdp1, Fdp2, Fdmu2])
        hess = np.array([Fddx, Fddmu, Fddvar, Fddtheta, Fddp1, Fddp2, Fddmu2])

        return grad, hess 

    def update(self, u, X, grad, hess):
        x2 = X[4]
        sigma_x = self.sigmoid(x2)
        sigma_dx = self.sigmoid_dx(x2)
        sigma_dxx = self.sigmoid_dxx(x2)

        
        grad[4] +=  -u/(sigma_x)*sigma_dx+(1-u)/(1-sigma_x)*sigma_dx
        
        hess[4] += -u/sigma_x*sigma_dxx +u/sigma_x**2*(sigma_dx**2) + \
                (1-u)/(1-sigma_x)*sigma_dxx + (1-u)/(1-sigma_x)**2*sigma_dx**2
        
        for i in range(hess.shape[0]):
            hess[i] = self.relu_abs(hess[i])
        
        X_new = X - np.nan_to_num(1/hess)*grad
        return X_new

    def update_response(self, u, X, grad, hess):
        x2 = X[9]
        sigma_x = self.sigmoid(x2)
        sigma_dx = self.sigmoid_dx(x2)
        sigma_dxx = self.sigmoid_dxx(x2)

        
        grad[9] +=  -u/(sigma_x)*sigma_dx+(1-u)/(1-sigma_x)*sigma_dx
        
        hess[9] += -u/sigma_x*sigma_dxx +u/sigma_x**2*(sigma_dx**2) + \
                (1-u)/(1-sigma_x)*sigma_dxx + (1-u)/(1-sigma_x)**2*sigma_dx**2
        
        for i in range(hess.shape[0]):
            hess[i] = self.relu_abs(hess[i])
        
        X_new = X - 1/hess*grad
        return X_new