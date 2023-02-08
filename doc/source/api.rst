API documentation
=================

Saltype datatypes
-----------------

Salt
^^^^
.. autoclass:: saltype.datatype.Salt
    :members: ALLOW_MIX_FLOAT, FLOAT_CACHE_MAX, invalidate, select, value, plain, recalc

Leaf
^^^^

.. autoclass:: saltype.datatype.Leaf
    :show-inheritance:
    :members: __init__, value

Tools
-----

SaltArray
^^^^^^^^^
.. autoclass:: saltype.tools.SaltArray
    :members: __init__, invalidate, invalidate_container,
        value, value_container, recalc, recalc_container,
        append, extend

Derivative
^^^^^^^^^^
.. autoclass:: saltype.tools.Derivative
    :show-inheritance:
    :members: __init__

sparse_derivative
^^^^^^^^^^^^^^^^^
.. autofunction:: saltype.tools.sparse_derivative

dump
^^^^
.. autofunction:: saltype.tools.dump

simplify
^^^^^^^^
.. autofunction:: saltype.tools.simplify

Empanada and Empanadiña
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: saltype.tools.empanada

.. autofunction:: saltype.tools.empanadina
