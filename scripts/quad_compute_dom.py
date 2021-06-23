#!/usr/bin/env python

import numpy as np
from scipy_wilson import Wilson
import pymc3 as pm
import reciprocalspaceship as rs
from scipy.stats.distributions import gamma,norm


#offFN = "/home/kmdalton/xtal/gfp/300fs_wo144/integration/xtals/abismal/checkpoint_0_300.mtz"
#onFN =  "/home/kmdalton/xtal/gfp/300fs_wo144/integration/xtals/abismal/checkpoint_1_300.mtz"
#phase_FN = "/home/kmdalton/xtal/gfp/models/GFP_SSRL_refine_54_final_P21.mtz"
offFN = "/home/kmdalton/opt/careless-examples/pyp/merge/pyp_0.mtz"
onFN  = "/home/kmdalton/opt/careless-examples/pyp/merge/pyp_1.mtz"
phase_FN = "/home/kmdalton/opt/careless-examples/pyp/phenix/PYP_refine_1.mtz"

outFN = "test.mtz"

nproc=1
chain_length = 3_000
burnin = 3_000

ds = rs.read_mtz(onFN).join(
    rs.read_mtz(offFN).expand_to_p1().expand_anomalous(), 
    rsuffix='off', 
    lsuffix='on', 
    check_isomorphous=False
).dropna()

ds = ds.compute_multiplicity().compute_dHKL().label_centrics()


Fon,SigFon,Foff,SigFoff,epsilon,dHKL = ds[[
    'Fon', 'SigFon', 
    'Foff', 'SigFoff', 
    'EPSILON', 'dHKL'
]].to_numpy('float').T
Centric = ds['CENTRIC'].to_numpy('bool')

Ion  = SigFon**2. + Fon**2.
Ioff = SigFoff**2. + Foff**2.
SigIon  = 2.*Fon*SigFon
SigIoff = 2.*Foff*SigFoff

from reciprocalspaceship.algorithms.scale_merged_intensities import mean_intensity_by_resolution,mean_intensity_by_miller_index
Sigmaon  = mean_intensity_by_resolution( Ion/epsilon, dHKL)
Sigmaoff = mean_intensity_by_resolution(Ioff/epsilon, dHKL)
Sigmamean = (Sigmaon + Sigmaoff)/2.


def sf_quadrature(Fon, SigFon, Foff, SigFoff, centric, Sigmaon, Sigmaoff, npoints=100):
    grid, weights = np.polynomial.chebyshev.chebgauss(npoints)
    logpon  = Wilson.structure_factor(centric[...,None], epsilon[...,None], Sigmaon[...,None]).logpdf
    logpoff = Wilson.structure_factor(centric[...,None], epsilon[...,None], Sigmaoff[...,None]).logpdf

#Integration window based on the normal, likelihood distribution
    window_size = 20. #In standard devs
    Jmin = Fon - window_size*SigFon/2.
    Jmin = np.maximum(0., Jmin)
    Jmax = Jmin + window_size*SigFon

    Kmin = Foff - window_size*SigFoff/2.
    Kmin = np.maximum(0., Kmin)
    Kmax = Kmin + window_size*SigFoff

    #Change of variables for generalizing chebgauss
    # Equivalent to: logweights = log(sqrt(1-grid**2.)*w)
    weights = np.sqrt(1. - grid**2.)*weights
    logweights = (0.5*(np.log(1-grid) + np.log(1+grid)) + np.log(weights))
    J = (Jmax - Jmin)[:,None] * grid / 2. + (Jmax + Jmin)[:,None] / 2.
    K = (Kmax - Kmin)[:,None] * grid / 2. + (Kmax + Kmin)[:,None] / 2.

    logPJ =  logpon(J)
    logPK = logpoff(K)

    logLJ = norm.logpdf(J,  Fon[...,None],  SigFon[...,None])
    logLK = norm.logpdf(K, Foff[...,None], SigFoff[...,None])
    
    DF = (
            (J[...,:,None] - K[...,None,:]) * \
            np.exp(
                logPJ[...,:,None] + logPK[...,None,:] + \
                logLJ[...,:,None] + logLK[...,None,:] + \
                logweights[...,:,None] + logweights[...,None,:]
            )
        ).sum(-1).sum(-1)*(Jmax - Jmin)*(Kmax - Kmin) / 4.

    DF2 = (
            (J[...,:,None] - K[...,None,:])**2. * \
            np.exp(
                logPJ[...,:,None] + logPK[...,None,:] + \
                logLJ[...,:,None] + logLK[...,None,:] + \
                logweights[...,:,None] + logweights[...,None,:]
            )
        ).sum(-1).sum(-1)*(Jmax - Jmin)*(Kmax - Kmin) / 4.

    return DF, DF2 - DF**2.



def quadrature(Ion, SigIon, Ioff, SigIoff, centric, Sigmaon, Sigmaoff, npoints=100):
    grid, weights = np.polynomial.chebyshev.chebgauss(npoints)

    #Gamma prior params
    a = np.where(centric, 0.5, 1.)
    scaleon  = 1./np.where(Centric, 2.*Sigmaon , Sigmaon)
    scaleoff = 1./np.where(Centric, 2.*Sigmaoff, Sigmaoff)

    #Integration window based on the normal, likelihood distribution
    window_size = 20. #In standard devs
    Jmin = Ion - window_size*SigIon/2.
    Jmin = np.maximum(0., Jmin)
    Jmax = Jmin + window_size*SigIon

    Kmin = Ioff - window_size*SigIoff/2.
    Kmin = np.maximum(0., Kmin)
    Kmax = Kmin + window_size*SigIoff

    #Change of variables for generalizing chebgauss
    # Equivalent to: logweights = log(sqrt(1-grid**2.)*w)
    weights = np.sqrt(1. - grid**2.)*weights
    logweights = (0.5*(np.log(1-grid) + np.log(1+grid)) + np.log(weights))
    J = (Jmax - Jmin)[:,None] * grid / 2. + (Jmax + Jmin)[:,None] / 2.
    K = (Kmax - Kmin)[:,None] * grid / 2. + (Kmax + Kmin)[:,None] / 2.

    logPJ = gamma.logpdf(J, a[...,None], scale= scaleon[...,None])
    logPK = gamma.logpdf(K, a[...,None], scale=scaleoff[...,None])

    logLJ = norm.logpdf(J,  Ion[...,None],  SigIon[...,None])
    logLK = norm.logpdf(K, Ioff[...,None], SigIoff[...,None])
    
    DF = (
            (J[...,:,None]**2. - K[...,None,:]**2.) * \
            np.exp(
                logPJ[...,:,None] + logPK[...,None,:] + \
                logLJ[...,:,None] + logLK[...,None,:] + \
                logweights[...,:,None] + logweights[...,None,:]
            )
        ).sum(-1).sum(-1)*(Jmax - Jmin)*(Kmax - Kmin) / 4.

    DF2 = (
            (J[...,:,None]**2. - K[...,None,:]**2.)**2. * \
            np.exp(
                logPJ[...,:,None] + logPK[...,None,:] + \
                logLJ[...,:,None] + logLK[...,None,:] + \
                logweights[...,:,None] + logweights[...,None,:]
            )
        ).sum(-1).sum(-1)*(Jmax - Jmin)*(Kmax - Kmin) / 4.

    return DF, DF2 - DF**2.



npoints=100
#df_quad, vdf_quad = quadrature(
#    Ion,
#    SigIon,
#    Ioff,
#    SigIoff,
#    Centric,
#    Sigmaon,
#    Sigmaoff,
#    npoints,
#)

df_quad, vdf_quad = sf_quadrature(
    Fon,
    SigFon,
    Foff,
    SigFoff,
    Centric,
    Sigmaon,
    Sigmaoff,
    npoints,
)
df_raw = (ds.Fon - ds.Foff).to_numpy()


phases = rs.read_mtz(phase_FN)["PHIF-model"]
out = ds[['Fon', 'SigFon', 'Foff', 'SigFoff']].join(phases)
out['DF'] = df_quad 
out['DF'] = out.DF.astype('F')
out['SigDF'] = np.sqrt(vdf_quad)
out['SigDF'] = out.SigDF.astype('Q')
out['W'] = (1. + vdf_quad**2. / np.mean(vdf_quad**2.))**-1.
out['W'] = out.W.astype("W")
out = out.dropna()
out.write_mtz(outFN)

from IPython import embed
embed(colors='linux')
from sys import exit
exit()
