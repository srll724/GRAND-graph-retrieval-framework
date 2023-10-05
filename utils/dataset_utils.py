import functools
import numpy as np


def control_rng(method):
    """
    The argument `rng` must be passed as a keyword argument
    or not passed.
    """
    @functools.wraps(method)
    def _method_wrapper(self, *args, **kwargs):
        rng = kwargs.get("rng")
        if rng is None:
            rng = self.rng
        else:
            # assert isinstance(rng, np.random.Generator)
            assert isinstance(rng, np.random.RandomState)
        kwargs["rng"] = rng
        return method(self, *args, **kwargs)
    return _method_wrapper
