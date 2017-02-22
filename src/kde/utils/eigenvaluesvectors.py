import numpy as np


def _eig_C(data):
    raise NotImplementedError()


def _eig_Python(data):
    raise NotImplementedError()


def eig(data, implementation=_eig_Python):
    return implementation(data)