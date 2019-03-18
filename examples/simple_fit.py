import numpy as np

import zfit

normal_np = np.random.normal(loc=2., scale=3., size=10000)

obs = zfit.Space("x", limits=(-10, 10))

mu = zfit.Parameter("mu", 1., -5, 5)
sigma = zfit.Parameter("sigma", 3., 1, 10)
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

data = zfit.data.Data.from_numpy(obs=obs, array=normal_np)

nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

minimizer = zfit.minimize.MinuitMinimizer()
result = minimizer.minimize(nll)

param_errors = result.error()
