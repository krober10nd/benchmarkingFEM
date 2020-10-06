from firedrake import Constant, exp
import math


def RickerWavelet(t, freq=2, amp=1.0):
    """Time-varying source function"""
    # shift so full wavelet is developd
    t = t - 3 * (math.sqrt(6.0) / (math.pi * freq))
    return (
        amp
        * (1.0 - (1.0 / 2.0) * (2.0 * math.pi * freq) * (2.0 * math.pi * freq) * t * t)
        * math.exp(
            (-1.0 / 4.0) * (2.0 * math.pi * freq) * (2.0 * math.pi * freq) * t * t
        )
    )


def delta_expr(x0, x, y, z=None, sigma_x=2000.0):
    """Spatial function to apply source"""
    sigma_x = Constant(sigma_x)
    if z is None:
        return exp(-sigma_x * ((x - x0[0]) ** 2 + (y - x0[1]) ** 2))
    else:
        return exp(-sigma_x * ((x - x0[0]) ** 2 + (y - x0[1]) ** 2 + (z - x0[2]) ** 2))
