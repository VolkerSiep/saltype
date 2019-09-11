API documentation
=================

SALT datatypes
--------------

Salt
^^^^
.. autoclass:: salt.datatype.Salt
    :members: ALLOW_MIX_FLOAT, FLOAT_CACHE_MAX, invalidate, select, value, plain, recalc
    
Leaf
^^^^

.. autoclass:: salt.datatype.Leaf
    :show-inheritance:
    :members: __init__, value
    
Tools
-----

SaltArray
^^^^^^^^^
.. autoclass:: salt.tools.SaltArray
    :members: __init__, invalidate, invalidate_container,
        value, value_container, recalc, recalc_container,
        append, extend
    
Derivative
^^^^^^^^^^
.. autoclass:: salt.tools.Derivative
    :show-inheritance:
    :members: __init__

sparse_derivative
^^^^^^^^^^^^^^^^^
.. autofunction:: salt.tools.sparse_derivative

dump
^^^^
.. autofunction:: salt.tools.dump

simplify
^^^^^^^^
.. autofunction:: salt.tools.simplify

Empanada and Empanadiña
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: salt.tools.empanada

.. autofunction:: salt.tools.empanadina
