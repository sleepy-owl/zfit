"""Baseclass for `Function`. Inherits from Model.

TODO(Mayou36): subclassing?
"""
#  Copyright (c) 2019 zfit

import typing

import zfit
from .basemodel import BaseModel
from .interfaces import ZfitFunc
from ..settings import ztypes
from ..util import ztyping


class BaseFunc(BaseModel, ZfitFunc):

    def __init__(self, obs=None, dtype: typing.Type = ztypes.float, name: str = "BaseFunc",
                 params: typing.Any = None):
        """TODO(docs): explain subclassing"""
        super().__init__(obs=obs, dtype=dtype, name=name, params=params)

    def _func_to_integrate(self, x: ztyping.XType):
        return self.func(x=x)

    def _func_to_sample_from(self, x):
        return self.func(x=x)

    # TODO(Mayou36): how to deal with copy properly?
    def copy(self, **override_params):
        new_params = self.params
        new_params.update(override_params)
        return type(self)(new_params)

    def gradients(self, x: ztyping.XType, norm_range: ztyping.LimitsType = None, params: ztyping.ParamsTypeOpt = None):
        # TODO(Mayou36): well, really needed... this gradient?
        raise NotImplementedError("What do you need? Use tf.gradient...")

    def as_pdf(self) -> "zfit.core.interfaces.ZfitPDF":
        """Create a PDF out of the function

        Returns:
            :py:class:`~zfit.core.interfaces.ZfitPDF`: a PDF with the current function as the unnormalized probability.
        """
        from zfit.core.operations import convert_func_to_pdf
        return convert_func_to_pdf(func=self)

    def _check_input_norm_range_default(self, norm_range, caller_name="", none_is_error=True):  # TODO(Mayou36): default

        return self._check_input_norm_range(norm_range=norm_range, caller_name=caller_name, none_is_error=none_is_error)
