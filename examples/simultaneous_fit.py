# create space
import zfit
import numpy as np

obs = zfit.Space("x", limits=(-10, 10))

mu_shared = zfit.Parameter("mu_shared", 1., -4, 6)
sigma_reso = zfit.Parameter("sigma_resonant", 1., 0.1, 10)
sigma_nonreso = zfit.Parameter("sigma_non_resonant", 1., 0.1, 10)

gauss_reso = zfit.pdf.Gauss(mu=mu_shared, sigma=sigma_reso, obs=obs)
gauss_nonreso = zfit.pdf.Gauss(mu=mu_shared, sigma=sigma_nonreso, obs=obs)

normal_np = np.random.normal(loc=2., scale=3., size=10000)
data_reso = zfit.data.Data.from_numpy(obs=obs, array=normal_np)

normal_np = np.random.normal(loc=2., scale=4., size=10000)
data_nonreso = zfit.data.Data.from_numpy(obs=obs, array=normal_np)

nll_simultaneous = zfit.loss.UnbinnedNLL(model=[gauss_reso, gauss_nonreso],
                                         data=[data_reso, data_nonreso])
# OR, equivalently
nll_reso = zfit.loss.UnbinnedNLL(model=gauss_reso, data=data_reso)
nll_nonreso = zfit.loss.UnbinnedNLL(model=gauss_nonreso, data=data_nonreso)
nll_simultaneous2 = nll_reso + nll_nonreso
