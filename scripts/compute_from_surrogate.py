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


Fon,SigFon,Foff,SigFoff = ds[[
    'Flocon', 'Fscaleon', 
    'Flocoff', 'Fscaleoff', 
]].to_numpy('float').T



def truncnorm_quadrature(Fon, SigFon, Foff, SigFoff, npoints=100, zero=0., inf=1e32):
    from scipy.stats import truncnorm
    grid, weights = np.polynomial.chebyshev.chebgauss(npoints)
    logQon  = truncnorm(zero, inf,  Fon[:,None], SigFon[:,None]).logpdf
    logQoff = truncnorm(zero, inf, Foff[:,None], SigFoff[:,None]).logpdf

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

    logQJ =  logQon(J)
    logQK = logQoff(K)

    
    DF = (
            (J[...,:,None] - K[...,None,:]) * \
            np.exp(
                logQJ[...,:,None] + logQK[...,None,:] + \
                logweights[...,:,None] + logweights[...,None,:]
            )
        ).sum(-1).sum(-1)*(Jmax - Jmin)*(Kmax - Kmin) / 4.

    DF2 = (
            (J[...,:,None] - K[...,None,:])**2. * \
            np.exp(
                logQJ[...,:,None] + logQK[...,None,:] + \
                logweights[...,:,None] + logweights[...,None,:]
            )
        ).sum(-1).sum(-1)*(Jmax - Jmin)*(Kmax - Kmin) / 4.

    return DF, DF2 - DF**2.




npoints=100
df_quad, vdf_quad = truncnorm_quadrature(
    Fon,
    SigFon,
    Foff,
    SigFoff,
)
df_raw = (ds.Fon - ds.Foff).to_numpy()


phases = rs.read_mtz(phase_FN)["PHIF-model"]
out = ds[['Fon', 'SigFon', 'Foff', 'SigFoff']].join(phases)
out['DF'] = df_raw
out['DF'] = out.DF.astype('F')
out['QDF'] = df_quad 
out['QDF'] = out.QDF.astype('F')
out['SigQDF'] = np.sqrt(vdf_quad)
out['SigQDF'] = out.SigQDF.astype('Q')
out['W'] = (1. + vdf_quad**2. / np.mean(vdf_quad**2.))**-1.
out['W'] = out.W.astype("W")
out = out.dropna()
out.write_mtz(outFN)

from IPython import embed
embed(colors='linux')
from sys import exit
exit()
