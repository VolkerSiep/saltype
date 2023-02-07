# -*- coding: utf-8 -*-

"""
Main package file of the saltype project. Please refer to documentation written
in the particular modules.
"""
from .datatype import (Leaf, Salt, sin, cos, tan, asin, acos, log10,
                       atan, sinh, cosh, tanh, sqrt, exp, log, inv, squ)

from .tools import (Derivative, simplify, dump, empanada, empanadina,
                    SaltArray, sparse_derivative, SaltDict)
