from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative
import scipy.integrate as integrate
import timeit
import math
from sklearn.metrics import r2_score
import lmfit

hat = fits.open('hat-p-7_b/hat-p-7_b.fits')


PRIMARY = hat['PRIMARY']

TCE_1 = hat['TCE_1']
table_tce1 = Table.read(TCE_1) #¿se puede hacer un array de BinTablesHDU?


tce1_time = table_tce1['TIME'].data
tce1_timecorr = table_tce1['TIMECORR'].data
# tce1_cadenceno = table_tce1['CADENCENO'].data
# tce1_phase = table_tce1['PHASE'].data
tce1_lc_init = table_tce1['LC_INIT'].data
tce1_lc_init_err = table_tce1['LC_INIT_ERR'].data
tce1_lc_white = table_tce1['LC_WHITE'].data
tce1_lc_detrend = table_tce1['LC_DETREND'].data
tce1_model_init = table_tce1['MODEL_INIT'].data
tce1_model_white = table_tce1['MODEL_WHITE'].data

t_data = 1 + np.delete(tce1_time[0:200], np.argwhere(np.isnan(tce1_lc_init[0:200]))[0:], axis=None) #take away the time values where we have a nan value in lc_init
lc_init_data = np.delete(tce1_lc_init[0:200], np.argwhere(np.isnan(tce1_lc_init[0:200]))[0:], axis=None) #take away the nan values in the data array of lc_init


def L(r, P, Z): #it is already multiplied by r² as it is in the derivate into the integral
    '''
    Obstruction function
    '''
    p = P/r
    z0 = Z/r
    z = np.zeros(len(z0)) 
    for i in range(len(z0)):
        # Reflects the information w.r.t. the y axis  
        if z0[i]>0:
            z[i] = z0[i]
        if z0[i]<0:
            z[i] = -z0[i]
        
        if z[i]>1+p:
            return 0.
        elif (abs(1-p)<z[i] and z[i]<=1+p):
            k0 = np.arccos((p**2 + z[i]**2 -1)/(2*p*z[i]))
            k1 = np.arccos((1-p**2 + z[i]**2)/(2*z[i]))
            L0 = k0*p**2
            L2 = np.sqrt((4*z[i]**2- (1+z[i]**2-p**2)**2)/4) 
            return (L0 + k1 - L2)*r**2/np.pi
        elif (z[i]<=1-p):
            return p**2*r**2
        elif z[i]<= p-1: 
            return 1.*r**2

def L_noarrays(r, P, Z): #it is already multiplied by r² as it is in the derivate into the integral
    '''
    Obstruction function
    '''
    p = P/r
    z0 = Z/r
    # Reflects the information w.r.t. the y axis  
    if z0>0:
        z = z0
    if z0<0:
        z = -z0

    if z>1+p:
        return 0.
    elif (abs(1-p)<z and z<=1+p):
        k0 = np.arccos((p**2 + z**2 -1)/(2*p*z))
        k1 = np.arccos((1-p**2 + z**2)/(2*z))
        L0 = k0*p**2
        L2 = np.sqrt((4*z**2- (1+z**2-p**2)**2)/4) 
        return (L0 + k1 - L2)*r**2/np.pi
    elif (z<=1-p):
        return p**2*r**2
    elif z<= p-1: 
        return 1.*r**2


def I_function(r, gamma1, gamma2):
  '''
  Quadratic limb-darkening function
  '''
  mu = np.sqrt(1-r**2) 
  return 1. - gamma1*(1-mu) - gamma2*(1-mu)**2


def integrand_1(r, t, p, a, b, T, delta, gamma1, gamma2):
    '''
    Integrand in the numerator
    '''

    #     p = 0.1 #radius ratio
    #     b = 0.7 #impact. parameter
    #     gamma1 = 0.296 #linear limb darkening 
    #     gamma2 = 0.34 #quadratic limb darkening
    #     a = 2. #normalized semi-major axis (normalized with the star radius)
    #     T = 10. #orbital period
    #     delta = 0. #orbital phase
    omega = 2*np.pi/T #angular velocity
    x = a*np.cos(omega*t+delta)
    z = np.sqrt(x**2 + b**2)
    T1 = derivative(L, r, dx=1e-6, args=(p,z))
    T2 = I_function(r, gamma1, gamma2)
    return T1*T2


def integrand_1_noarrays(r, t, p, a, b, T, delta, gamma1, gamma2):
    '''
    Integrand in the numerator
    '''

    #     p = 0.1 #radius ratio
    #     b = 0.7 #impact. parameter
    #     gamma1 = 0.296 #linear limb darkening 
    #     gamma2 = 0.34 #quadratic limb darkening
    #     a = 2. #normalized semi-major axis (normalized with the star radius)
    #     T = 10. #orbital period
    #     delta = 0. #orbital phase
    omega = 2*np.pi/T #angular velocity
    x = a*np.cos(omega*t+delta)
    z = np.sqrt(x**2 + b**2)
    T1 = derivative(L_noarrays, r, dx=1e-6, args=(p,z))
    T2 = I_function(r, gamma1, gamma2)
    return T1*T2



def integrand_2(r, gamma1, gamma2):
    '''
    Integrand in the denominator
    '''
    return I_function(r, gamma1, gamma2)*2*r


def lc_fit(t, p, a, b, T, delta, gamma1, gamma2):
    
    # Main Loop to calculate the Flux

    Integral_1 = integrate.quad(integrand_1, 0.0001, 1., args=(t, p, a, b, T, delta, gamma1, gamma2))[0]
    Integral_2 = integrate.quad(integrand_2, 0.0001, 1., args=(gamma1, gamma2))[0]
    F= Integral_1/Integral_2
    return F

def lc_fit_noarrays(t, p, a, b, T, delta, gamma1, gamma2):
    
    # Main Loop to calculate the Flux

    Integral_1 = integrate.quad(integrand_1_noarrays, 0.0001, 1., args=(t, p, a, b, T, delta, gamma1, gamma2))[0]
    Integral_2 = integrate.quad(integrand_2, 0.0001, 1., args=(gamma1, gamma2))[0]
    F= Integral_1/Integral_2
    return F


def lc_resid(params, t, lc_data):
    p = params['p'].value
    a = params['a'].value
    b = params['b'].value
    T = params['T'].value
    delta = params['delta'].value
    gamma1 = params['gamma1'].value
    gamma2 = params['gamma2'].value
    
    Integral_1 = np.zeros(len(t))
    Integral_2 = np.zeros(len(t))
    F = np.zeros(len(t))
    
    for i in range(len(t)):
        Integral_1[i] = integrate.quad(integrand_1, 0.0001, 1., args=(t, p, a, b, T, delta, gamma1, gamma2))[0]
        Integral_2[i] = integrate.quad(integrand_2, 0.0001, 1., args=(gamma1, gamma2))[0]
        F[i]= 1. - Integral_1[i]/Integral_2[i]
    
    return F - lc_data


def main():
    params = lmfit.Parameters()
    params.add(name ='p', value = 1.0)
    params.add(name ='a', value = 1.0)
    params.add(name ='b', value = 1.0)
    params.add(name ='T', value = 1.0)
    params.add(name ='delta', value = 1.0)
    params.add(name ='gamma1', value = 1.0)
    params.add(name ='gamma2', value = 1.0)

    fit = lmfit.minimize(lc_resid, params, args =(t_data, lc_init_data))
    p = 1.45921258
    a = 1.10078824
    b = 0.44740415
    T = 0.99944628
    delta = 1.11166236
    gamma1 = 79.0120679
    gamma2 = -53.5736054

    v_lc = np.vectorize(lc_fit_noarrays, excluded=set([1]))

    y_model = v_lc(t_data, p, a, b, T, delta, gamma1, gamma2)
    y_model
    R_squared = r2_score(lc_init_data, y_model)

    print(lmfit.report_fit(fit))
    print(R_squared)

if __name__ == "__main__":
    main()