import numpy as np

SDR_DIFFUSION_WHITE = 203
HLG_LUMINANCE_RANGE = 1000
PQ_LUMINANCE_RANGE = 10000


def Gamma22Oetf(x):
    x = np.clip(x, 0, 1)
    return x ** (1/2.2)


def Gamma22Eotf(x):
    x = np.clip(x, 0, 1)
    return x ** 2.2


def PqOetf(x):
    m1 = 2610 / 16384
    m2 = (2523 / 4096) * 128
    c1 = 3424 / 4096
    c2 = (2413 / 4096) * 32
    c3 = (2392 / 4096) * 32
    x = np.clip(x, 0, 1)
    temp = x ** m1
    numerator = c1 + c2 * temp
    denominator = 1 + c3 * temp
    return np.clip((numerator / denominator) ** m2, 0, 1)


def PqEotf(x):
    m1 = 2610 / 16384
    m2 = (2523 / 4096) * 128
    c1 = 3424 / 4096
    c2 = (2413 / 4096) * 32
    c3 = (2392 / 4096) * 32
    x = np.clip(x, 0, 1)
    temp = x ** (1/m2)
    numerator = np.clip(temp-c1, 0, None)
    denominator = c2 - c3 * temp
    return np.clip((numerator/denominator)**(1/m1), 0, 1)