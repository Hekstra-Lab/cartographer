from scipy.stats import gengamma
import numpy as np


class Wilson():
    @staticmethod
    def structure_factor(centric, epsilon, Sigma):
        scale = np.where(centric, np.sqrt(2*Sigma*epsilon), np.sqrt(Sigma*epsilon))
        c = 2.
        a = np.where(centric, 1/2., 1.)
        return gengamma(a, c, loc=0., scale=scale)

    @staticmethod
    def intensity(centric, epsilon, Sigma):
        scale = np.where(centric, 2*Sigma*epsilon, Sigma*epsilon)
        c = 2.
        a = np.where(centric, 1/2., 1.)
        return gengamma(a, c, loc=0., scale=scale)

if __name__=="__main__":
    n = 10
    epsilon = np.random.choice([1., 2, 3, 4, 6], n)
    Sigma = 10.*np.random.random(size=n)
    F = np.linspace(0., 10., 1000)

    #First we test centric structure factors
    expected = np.sqrt(2/np.pi/Sigma[:,None]/epsilon[:,None]) * np.exp(-F[None,:]**2./2/Sigma[:,None]/epsilon[:,None])
    test = Wilson.structure_factor(True, epsilon[:,None], Sigma[:,None]).pdf(F)
    assert np.allclose(expected, test)

    #Now we test acentric structure factors
    # P(F) = 2*F/(Sigma*epsilon) * exp(-F**2./Sigma/epsilon)
    expected = 2*F[None,:]/Sigma[:,None]/epsilon[:,None] * np.exp(-F[None,:]**2./Sigma[:,None]/epsilon[:,None])
    test = Wilson.structure_factor(False, epsilon[:,None], Sigma[:,None]).pdf(F)

    assert np.allclose(expected, test)

    from IPython import embed
    embed()
