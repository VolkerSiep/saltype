API documentation
=================

SALT datatypes
--------------

Salt
^^^^
.. autoclass:: salty.datatype.Salt
    :members: ALLOW_MIX_FLOAT, FLOAT_CACHE_MAX, invalidate, select, value, plain, recalc
    
Leaf
^^^^

.. autoclass:: salty.datatype.Leaf
    :show-inheritance:
    :members: __init__, value
    
Tools
-----

SaltArray
^^^^^^^^^
.. autoclass:: salty.tools.SaltArray
    :members: __init__, invalidate, invalidate_container,
        value, value_container, recalc, recalc_container,
        append, extend
    
Derivative
^^^^^^^^^^
.. autoclass:: salty.tools.Derivative
    :show-inheritance:
    :members: __init__

sparse_derivative
^^^^^^^^^^^^^^^^^
.. autofunction:: salty.tools.sparse_derivative

dump
^^^^
.. autofunction:: salty.tools.dump

simplify
^^^^^^^^
.. autofunction:: salty.tools.simplify

Empanada and Empanadiña
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: salty.tools.empanada

.. autofunction:: salty.tools.empanadina
