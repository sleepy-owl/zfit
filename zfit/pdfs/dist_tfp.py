"""
A rich selection of analytically implemented Distributions (pdfs) are available in
`TensorFlow Probability <https://github.com/tensorflow/probability>`_. While their API is slightly
different from the zfit pdfs, it is similar enough to be easily wrapped.

Therefore a convenient wrapper as well as a lot of implementations are provided.
"""

import tensorflow_probability as tfp

import tensorflow as tf

from zfit import ztf
from zfit.core.basepdf import BasePDF
from zfit.core.limits import no_norm_range, supports


class WrapDistribution(BasePDF):  # TODO: extend functionality of wrapper, like icdf
    """Baseclass to wrap tensorflow-probability distributions automatically.

    """

    def __init__(self, distribution, name=None, **kwargs):
        # Check if subclass of distribution?
        name = name or distribution.name
        kwargs.update({k: v for k, v in distribution.parameters.items() if isinstance(v, tf.Variable)})

        super(WrapDistribution, self).__init__(name=name, **kwargs)
        # self.tf_distribution = self.parameters['distribution']
        self.tf_distribution = distribution

    def _unnormalized_prob(self, x):
        return self.tf_distribution.prob(value=x, name="unnormalized_prob")  # TODO name

    # TODO: register integral
    @supports()
    def _analytic_integrate(self, limits, norm_range):
        lower, upper = limits.get_boundaries()
        lower = ztf.to_real(lower[0])
        upper = ztf.to_real(upper[0])
        integral = self.tf_distribution.cdf(upper) - self.tf_distribution.cdf(lower)
        return integral


class Normal(WrapDistribution):
    def __init__(self, mu, sigma, name="Normal"):
        distribution = tfp.distributions.Normal(loc=mu, scale=sigma, name=name + "_tf")
        super().__init__(distribution=distribution, name=name)