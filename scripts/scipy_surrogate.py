from tensorflow_probability import distributions as tfd
from scipy.stats.distributions import truncnorm
import numpy as np


class TruncNorm():
    @staticmethod
    def posterior(loc, scale, low=0., high=1e32):
        return truncnorm(low, high, loc, scale)

if __name__=="__main__":

    X = np.linspace(0., 10., 1000)
    n = 10
    loc,scale = np.random.random(size=(2, n))

    expected = tfd.TruncatedNormal(loc[:,None], scale[:,None], 0., 1e32).log_prob(X).numpy()
    test = TruncNorm.posterior(loc[:,None], scale[:,None]).logpdf(X)
    

    from IPython import embed
    embed()
