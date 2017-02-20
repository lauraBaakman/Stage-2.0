import numpy as np

def _covariance_C(data):
    raise NotImplementedError()

def _covariance_Python(data):
    raise NotImplementedError()

def covariance(data, implementation=_covariance_Python):
    return implementation(data)

