#!/usr/bin/env python

import numpy as np
import pymc3 as pm
import reciprocalspaceship as rs


offFN = "/home/kmdalton/xtal/gfp/300fs_wo144/integration/xtals/abismal/checkpoint_0_300.mtz"
onFN =  "/home/kmdalton/xtal/gfp/300fs_wo144/integration/xtals/abismal/checkpoint_1_300.mtz"
phase_FN = "/home/kmdalton/xtal/gfp/models/GFP_SSRL_refine_54_final_P21.mtz"
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


Fon,SigFon,Foff,SigFoff,epsilon,dHKL = ds[['Fon', 'SigFon', 'Foff', 'SigFoff', 'EPSILON', 'dHKL']].to_numpy('float').T
Centric = ds['CENTRIC'].to_numpy('bool')

from reciprocalspaceship.algorithms.scale_merged_intensities import mean_intensity_by_resolution
Sigmaon  = mean_intensity_by_resolution( Fon*epsilon**-0.5, dHKL)
Sigmaoff = mean_intensity_by_resolution(Foff*epsilon**-0.5, dHKL)


#Gamma prior params
a = np.where(Centric, 0.5, 1.)
scaleon  = 1./np.where(Centric, 2.*Sigmaon , Sigmaon)
scaleoff = 1./np.where(Centric, 2.*Sigmaoff, Sigmaoff)

with pm.Model() as model:
    Ion  = pm.distributions.Gamma('Ion',  a,  scaleon, shape=len(a))
    Ioff = pm.distributions.Gamma('Ioff', a, scaleoff, shape=len(a))
    Fon  = (epsilon*Ion)**0.5
    Foff = (epsilon*Ioff)**0.5
    DeltaF = pm.Deterministic("DeltaF", Fon - Foff)
    likelihood_on  = pm.distributions.Normal('Likelihood_on' ,  mu=Fon,  sigma=SigFon, observed=Fon)
    likelihood_off = pm.distributions.Normal('Likelihood_off', mu=Foff, sigma=SigFoff, observed=Foff)
    trace = pm.sample(draws=chain_length, tune=burnin, cores=nproc)

samples = trace.get_values('DeltaF')

phases = rs.read_mtz(phase_FN)["PHIF-model"]
out = ds[['Fon', 'SigFon', 'Foff', 'SigFoff']].join(phases)
out.join(phases)
out['DF'] = samples.mean(0)
out['SigDF'] = samples.std(0)
out['DF'] = out['DF'].astype('F')
out['SigDF'] = out['SigDF'].astype('Q')

out = out.dropna()

from IPython import embed
embed(colors='linux')
from sys import exit
exit()
