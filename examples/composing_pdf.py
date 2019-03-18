import zfit

obs = zfit.Space("x", limits=(-10, 10))

mu = zfit.Parameter("mu", 1, -4, 6)
sigma = zfit.Parameter("sigma", 1, 0.1, 10)
lambd = zfit.Parameter("lambda", -1, -5, 0)
frac = zfit.Parameter("fraction", 0.5, 0, 1)

gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
exponential = zfit.pdf.Exponential(lambd, obs=obs)

sum_pdf = zfit.pdf.SumPDF([gauss, exponential], fracs=frac)

sum_pdf = frac * gauss + exponential
