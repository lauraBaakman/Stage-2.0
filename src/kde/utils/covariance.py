import numpy as np

def _covariance_C(data):
    raise NotImplementedError()

def _covariance_Python(data):
    return np.cov(data.transpose(), bias=True)

def covariance(data, implementation=_covariance_Python):
    return implementation(data)

