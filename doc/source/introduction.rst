Introduction
============

Background and objective
------------------------

**saltype** stands for **S**\ ymbolic **A**\ lgebra **L**\ ight **Type**, and is also
a common additive in food preparation, which makes the name fit in the palette
of software developed by me. Salt (as NaCl) is also a substance that can be
met in high quantities in nature. This can be seen in parallel to *SALT*
being developed to obtain large-scale derivatives.

Without compromise, this package is developed to be small and efficient
solely for the purpose of obtaining symbolic derivatives of native-looking
python code. I called the first version of this code *sympy*, but others
made a much bigger, more general, and for my purposes too slow :-) package
with that very same name: Sympy_. To call a symbolic package in python
*sympy* is of course not a great act of creativity, so this didn't come
as a big surprise.

*saltype* does hardly anything of what Sympy_ does, even if some of the
functionality could be included of course, but this is not the objective of
*saltype*. The closest package in terms of objective currently known to me
is CasADi_, and the basic syntax and approach is similar.

.. _CasADi: https://github.com/casadi
.. _Sympy: http://www.sympy.org

CasADi_ is again a bigger and embracing package, and with no doubt way
more advanced than *saltype*. Its documentation states for the following
interesting benchmark as CasADi_ code::

    from casadi import *
    x = SX.sym("x")
    y = x
    for i in range(100):
       y = sin(y) * y

"*In Casadi, this code executes in the blink of a second, whereas
conventional computer algebra systems fail to construct the expression
altogether.*"

A direct comparison has not been done here, but the following
native python code with *saltype* executes in 0.6 ms
(excluding the import statement)::

    from salt import Leaf, sin
    y = x = Leaf(3.14159)
    for _ in range(100):
        y = sin(y) * y

... and it takes 2 ms to derive that equation with respect to *x* and 0.5 ms
to evaluate the derivative::

    from salt import Derivative
    z = Derivative([y], [x])[0,0]  # derive: z = dy/dx

    z.invalidate()  # only necessary if value of x changed
    print z.value  # re-evaluate value after invalidate

As with CasADi_, the runtime increases linearly with problem size, that is
for instance the :code:`100` in the :code:`xrange` statement. The performance
is less than an order of magnitude slower than CasADi_, but this is not a big
shame given that *saltype* is purely python.

A noticeable difference is that *salty* does not use names (as strings) for
the symbols, as they are not needed for the intended purpose. Furthermore,
the derivative algorithm is symbolic and with that rather primitive compared
to the advanced application of forward and reverse algorithmic differentiation
as implemented in CasADi_.

**The main reason to maintain *saltype* aside is to hold a lightweight package
(currently 50 kB source files, of which most of it is inline documentation)
that is almost not dependent on anything but python itself (I need the ``numpy``
float datatype in order not to throw exceptions for evaluations yielding ``nan``
or ``inf``. I developed *saltype* to create and evaluate Jacobian matrices of
systems with up to several thousand variables - as efficient and pythonic
as possible**.

Key concept
-----------
Each algebraic operation is defined as a node in a directed graph, whereas the
edges of the graph represent the dependencies between nodes. For the expression
``c = a + b``, ``c`` is a node of type ``+``, whereas the types of ``a`` and
``b`` are determined by their definition. Node ``c`` points to ``a`` and ``b``,
but not vice versa. *SALT* supports the following node types:

**Leaf (source) nodes**
  These nodes are not dependent on any *child* nodes (operands), but contain
  a value that can be changed by the client code.

**Zero and One and constant nodes**
  These nodes also have no *child* nodes, and their value is fixed to zero,
  one, or any value respectively. In particular automatically derived code
  can easily be simplified if these entities are explicitly identified.
  Other used constants are also cached to some degree, mainly to identify
  duplicates for simplification purposes.

**Unary nodes**
    These represent all unary mathematical functions, such as trigonometric
    functions, *log*, *exp*, *sqrt*, etc. They have one *child* node,
    representing their argument. Additional nodes are defined for *1/x* and
    *x\*\*2* for more efficiency, called *inv* and *squ* respectively.

**Operator nodes**
    These are individually implemented nodes to represent all binary
    operators (+, -, \*, /, \*\*) as well as unary minus and a primitive
    selector (decision) node that evaluates to::

        x.select(y) := 0 if y<=0 else x

The entire data structure of a node consists of

1. A type identifier
2. A cached scalar float value
3. A reference counter
4. A fixed size list of operands (child nodes)

This minimal structure allows to treat large quantities of symbols.
In particular, without requiring bidirectional links, the resulting fixed-size
record avoids dynamic memory allocation and can on demand easily be implemented
in *C* as a python extension for further performance gain, if that is one day
necessary. But to place the 853862\ :sup:`th` quotation of the following::

    # Premature optimization is the root of all evil
    # (or at least most of it) in programming.
    #           -- Computer Programming as an Art (1974), Donald E. Knuth

Expressions are simplified at earliest possible stage. The code::

    y = sqrt(x) * sqrt(x)

would be instantaneously simplified to::

    y = squ(sqrt(x)) = x

Typical application and workflow
--------------------------------
Graph generation
^^^^^^^^^^^^^^^^
The client code instantiates a number of :py:obj:`Leaf <saltype.datatype.Leaf>`
objects (independent variables). The subsequent procedural code defines
the graph, while its procedural nature guaranties the graph to be acyclic.

The user-visible datatype is :py:obj:`Salt <saltype.datatype.Salt>`, being
the base-class of :py:obj:`Leaf <saltype.datatype.Leaf>`.
It behaves very similar to the built-in ``float`` type with one major
exception, that is the non-existence of comparison operators. We cannot compare
the value of symbols at graph generation time, as their value is dynamic.

The procedural code can be part of any *python* language construct, including
loops, functions, recursions and classes. It can also be part of container types,
due to the mutable nature though not as keys in dictionaries or as items in sets.
The :py:obj:`Salt <saltype.datatype.Salt>` datatype is a smart-pointer
to the node objects (with reference counting) and defines the convenience
operators and functions to give the (almost) full ``float`` experience.

At the end of this phase, the client code has obtained the *dependent*
variables, thus both *independent* and *dependent* variables are now
available as :py:obj:`Salt <saltype.datatype.Salt>` objects.

A small example without any practical justification is::

    from saltype import Leaf, sin, cos, log

    x1, x2 = map(Leaf, 2.5, 0.1)
    a = x1 * cos(x2)
    b = x1 * sin(x2)
    y1 = sqrt(a) + log(b)
    y2 = y1 * b

    x = [x1, x2] # independent variables
    y = [y1, y2] # dependent variables


Repeated evaluation
^^^^^^^^^^^^^^^^^^^
Given above code, we can now re-evaluate the dependent variables for different
values of the independent variables. To do so, the dependent variables
are marked as invalid, and the new values are set to the independent ones.
Afterwards, the new values of the dependent variables can be queried::

    while nobody_is_bored:
        y1.invalidate()
        y2.invalidate()

        x1.value = 2.0 # in real application of course ...
        x2.value = 0.2 #   ... non-constant assignments

        print y1.value, y2.value # ... and processing of these

The step calling ``invalidate`` seems nasty, but is a small price for not requiring
bidirectional links between the nodes - with all disadvantages that would
yield.

Generating derivatives
^^^^^^^^^^^^^^^^^^^^^^
For optimisation, equation solving, and other exercises of this kind,
the derivatives *dy/dx* are more than welcome. The ability to derive equations
is *my* entire motivation to use symbolic algebra::

    z = Derivative(y, x)
    simplify(z)

The derivative algorithm already performs the same simplifications as applied
by the graph generation phase. In the explicit *in-place* ``simplify`` call,
common terms are identified and simplified to be represented only once, for
instance::

    y = sin(a + b) * cos(b + a)

will be simplified to::

    var_1 = a + b
    y = sin(var_1) * cos(var_1)

This elimination of duplicates is essential to generate efficient derivatives
and might in future versions well be included into the
:py:obj:`Derivative <saltype.tools.Derivative>` class.

Normally, the generated derivative symbols undergo the same repeated evaluation
as the dependent variables. Consequently, higher order derivatives are naturally
supported, as long as the exponential growths of symbols required to
represent higher order derivatives can be handled in memory. You would probably not
want to take the 5th derivative of an 800 times 800 system.

Advanced topics
===============
There are not many advanced topics to *saltype* as a main objective is to keep
things simple. Yet, there are some hidden peanuts:

Floats and Leafs
----------------
The python operator overloading in *saltype* makes it possible to smoothly mix
``float`` and :py:obj:`Salt <saltype.datatype.Salt>` data types. Naturally, the
symbolic graph is only built when using :py:obj:`Salt <saltype.datatype.Salt>`
entities. Consider ::

    a = Leaf(3.0)
    b = 4.0 * 2.0
    c = a + b

The ``+`` operator still converts ``b`` to a symbolic node before creating the
node representing ``c``, but this is an anonymous node with no user reference
to change its value later on - in contrast to ``a``. In the symbolic context,
``b`` can therefore be called a *constant*. Obviously, the information that ``b``
is the product of four and two is not preserved either.

Typical applications of such mixing for the sake of readability is::

    m = Leaf(75.0)  # kg
    v = Leaf(4.0)  # m/s
    E = 0.5 * m * squ(v)  # Energy of a person running

The alternative code with pure data types would look like (**don't do this for
the reason described below**)::

    m = Leaf(75.0) # kg
    v = Leaf(4.0) # m/s
    a_half = Leaf(0.5)
    E = a_half * m * squ(v) # Energy of a person running

Not only is this less readable or natural, but also can *saltype* in the latter
code not know whether the user intends later to change the value of ``a_half``.
For the upper code, *saltype* can recognise this and reuse nodes of the same value
in other expressions by caching. If you are to simulate the *Paris Marathon* with
50000 participants, the upper code would still only hold one reference to constant
node of value ``0.5``. Simplification could (*does not yet though*) multiply out
that factor when adding the energies, and reuse it when deriving the terms.

There is more::

    f = Leaf(20)  # frequency [f] = 1/sec
    t = 1 / f  # period [t] = sec

Above code will recognise ``1`` as the famous number *one* and simplify above equation to ::

    t = inv(f)

with a simpler derivative and more simplification chances when used further on.
This works, because the floats *zero*, *one* and *two* are pre-cached as the
special nodes dedicated to them.

As an exception, the :py:obj:`select <saltype.datatype.Salt.select>` method does not
accept ``float`` type arguments, just because it would never make sense.

.. seealso::

    Attributes :py:obj:`ALLOW_MIX_FLOAT <saltype.datatype.Salt.ALLOW_MIX_FLOAT>` and
    :py:obj:`FLOAT_CACHE_MAX <saltype.datatype.Salt.FLOAT_CACHE_MAX>`

.. _empanada_empanadina:

Empanadas and Empanadiñas
-------------------------
Empanada_ is a delicious wrap dish originating from Galicia in Spain,
coincidentally also the place where my wife grew up. Now, in this context,
it is a metaphor for the functionality to wrap your own *meat* into the
network (bread) of *saltype* symbolic algebra nodes. Empanadiñas are just
small Empanadas.

.. _Empanada: https://en.wikipedia.org/wiki/Empanada

Say you largely rely on *saltype* to generate the derivatives of the dependent
variables :math:`y` with respect to the independent ones :math:`x`, but for a
block of intermediate equations :math:`u(v)` with

.. math::
    v = v(x)\quad\text{and}\quad y = y(x, u)

a manual implementation of the derivatives :math:`\mathrm{d}u/\mathrm{d}v`
is desired. This can have several reasons:

  - You need to implement an existing subroutine that can only be evaluated
    with *float*, but on the other hand is capable of delivering its derivative.
  - A considerable part of the equations is more efficiently derived
    manually.

The concept of a ``plain`` operator enables this feature in an elegant, even
if probably not most efficient way, such that the outer derivative
:math:`\mathrm{d}u/\mathrm{d}v` still can be generated, and new values for
:math:`y` and :math:`\mathrm{d}y/\mathrm{d}x` can be evaluated
without having to consider the inclusion.

The ``plain`` operator :math:`\mathrm{plain}(x)` evaluates always to
:math:`x`, but we *forget* the dependencies when deriving, i.e.
:math:`\mathrm{d}p/\mathrm{d}x \equiv 0`. Now, this sounds like giving
a monkey a screw to open a banana, doesn't it!?

To explain this, we denote symbolic variables with an accent
:math:`\hat \psi`, and pure numerical variables without (:math:`\psi`)


Given :math:`u(v)` and :math:`J = \mathrm{d}u/\mathrm{d}v` as numerical values
from the *unSALTed* subroutine, define the symbols :math:`\hat u(\hat v)`
as a *Taylor* expansion:

.. math::
    \hat u = u(v) + J \cdot (\hat v - \mathrm{plain}(\hat v))

With multiple variables (that is, :math:`u` and :math:`v` are vectors),
:math:`J` is a matrix and the multiplication an inner product. This way,
the value and the first derivative of :math:`u` are correctly evaluated.
The series can be expanded in order to reproduce higher order derivatives -
though this is not supported by *Empanada* and *Empanadiña* I'm afraid.

For first order (derivative consistent) embedding however, the functionality is
implemented as the :py:obj:`empanada <saltype.tools.empanada>` function in general
and as the :py:obj:`empanadina <saltype.tools.empanadina>` function for scalar
functions.

Empanadiña example
^^^^^^^^^^^^^^^^^^
Consider the desire to embed the following (``float`` type) function into the
*saltype* symbolic graph::

    def func(x):
        y = x ** 6
        J = 6 * x ** 5
        return y, J

This is a scalar function that turns its argument :math:`x` into a function value
:math:`y`, also providing the manually implemented derivative
:math:`J = \frac{\mathrm{d} y}{\mathrm{d} x}`.

The following code wraps this function into the symbolic algebra graph::

    a = Leaf(2.0)
    b = sqrt(a)
    y = empanadina(func, b) # has the effect of "y = func(b)" in symbolic context

A subsequent :code:`dyda = Derivative([y], [a])[0,0]` will give the correct total
derivative :math:`\frac{\mathrm{d} y}{\mathrm{d} a} = J \cdot
\frac{\mathrm{d} b}{\mathrm{d} a}`.

Empanada example
^^^^^^^^^^^^^^^^
In most practical cases, the function to embed has either a vectorial input argument,
a vectorial return value, or both. The bigger sister of
:py:obj:`empanadina <saltype.tools.empanadina>`, namely :py:obj:`empanada <saltype.tools.empanada>`
is used in this case. Let the function now be::

    import math
    def func(x):
        a, b = x
        c = math.exp(a + b)
        y = [a, a * math.sin(b), c]
        J = [[1.0, 0.0],
             [math.sin(b), a * math.cos(b)],
             [c, c]]
        return y, J

The embedding is very similar to above example. We just need to tell the dimensionality
of the function result as ``dim_out``, because :py:obj:`empanada <saltype.tools.empanada>`
needs to prepare the symbols and would not like to call the function just to find it out::

    x = Leaf(2.0)
    z = [x * x, sqrt(x)]
    y = empanada(func, z, dim_out=3)

    dydx = Derivative(y, [x])

The current implementation of :py:obj:`empanadina <saltype.tools.empanadina>` is actually
only a wrapper of :py:obj:`empanada <saltype.tools.empanada>` to relieve the user from
cluttering indexing, like so::

    def empanadina(func, inp):
        def _func(inp):
            out, jac = func(inp[0])
            return [out], [[jac]]
        return empanada(_func, [inp])[0]

This might change in the future according to the plan to let ``empanadina`` embed
functions that deliver *n*\ :sup:`th` order derivatives.

Iterative algorithms
--------------------
The following thinking applies to all iterative algorithms, but is here exemplified with
the task of solving an implicit equation or equation system.

.. warning::
    Do not do the following - ever!

You might have the glorious idea to use *saltype* or any other symbolic algebra
system as follows in for instance a fixpoint iteration::

    # 1. solve for some fixpoint
    x = Leaf(start_value)
    while not converged and still_memory_left:
        dx = f(x, p)
        x = x + dx

    # 2. Obtain the derivative of x(p) with respect to p
    dxdp = Derivative([x], [p])

.. warning::
    Do not do the above - ever!

If you now think: "*Why not?*", please read on.


Here is what you might do instead::

    x = Leaf(start_value)
    y = f(x, p) # generate the function symbolically once!
    partial = Derivative([y], [x, p])[0] # take the derivative already
    dxdp = -partial[1] / partial[0] # magic equation, see below

    while not_converged:
        x.value += y.value # iterate on the graph, don't extend it
        y.invalidate() # don't forget to invalidate before re-evaluate

The magic assignment of `dxdp` represents the following mathematics:
We know the algorithm to terminate (if successful) with :math:`f(x, p)=0`.
The total differential gives the equation:

.. math::
    \left .\frac{\partial f}{\partial x}\right |_p \mathrm{d}x +
    \left .\frac{\partial f}{\partial p}\right |_x \mathrm{d}p = 0
    \quad\Rightarrow\quad
    \frac{\mathrm{d}x}{\mathrm{d}p} =
    -\left ( \left .\frac{\partial f}{\partial x}\right |_p \right )^{-1}\cdot
    \left .\frac{\partial f}{\partial p}\right |_x

And once you have the
derivative :math:`\partial f/\partial x|_p` at hand, you might just as well
use `Newton's method`_ to solve :math:`f(x, p)=0` instead of the primitive
fix-point iteration::

    x = Leaf(start_value)
    y = f(x, p)
    partial = Derivative([y], [x, p])[0]
    dx = -y / partial[0]

    while not_converged:
        x.value += dx.value
        dx.invalidate()

This works also perfectly for multi-variant systems.

.. _`Newton's method`: https://en.wikipedia.org/wiki/Newton's_method

Limitations
===========
Limitations can be a bad thing, but also prevent the user from doing stupid
things. In that sense, please see the following limitations as features.

Necessity of *invalidate*
-------------------------
I should be sorry for this one, but it is part of the key for the performance.

In a previous version of this package, nodes automatically send their invalidity
status upwards the graph whenever their value was set, until an already invalid
node was reached. This was convenient from a programmers' point of view.
Now, that I don't have it anymore, I myself find me frequently swearing when I
discover that I forgot to call ``invalidate`` again.

**But** the price for the automatic propagation of validity status upwards is a
bidirectional linking of nodes. Profiling my old package revealed that 99%
of the time was spent in memory-allocations to handle the dynamic lengths list
of node parent pointers - even and in particular after I desperately ported
the package to C. Note that frequently used nodes can have thousands of parents
within the symbolic graph.

Having written this, I play with the thought to follow another concept, namely to freeze a graph once all knitting, derivatives and simplifications are done. Freezing would install the upwards links (once and for all) and allow again automatic, slightly more efficient, and consistent invalidation. The drawback of this would be memory usage and the necessity to be strict in keeping frozen graphs imutable. Currently, I would not know how to enforce this at least half way elegantly.

Acyclicity
----------
Would it not be nice to allow cycles in the graph and that way encode
iterative algorithms? Or what about replacing existing nodes within the
graph with new ones? -- **Well you wish!**

The guaranteed non-circular nature of the symbolic graph is a required property
for efficient evaluation and construction of derivatives. If you need
iterations, please do that outside *saltype* (which is exactly the targeted
application) or use another package that provides such functionality.

Numpy and Scipy incompatibility
-------------------------------
Well, this one is not easy to sell as a feature, but as a fact, the full
numpy functionality is only accessible with a set of standard data types,
of which the *saltype* symbols are not one of them.

However, of course the result of what comes out of *saltype* in terms of
values is mostly meant to be processed by numpy, scipy and similar packages.

If you however find a native python numeric library, there is a chance that
*saltype* objects fit right in -- at least as long as nobody tries to use
comparison operators on the entities, as for instance to pivot a matrix for
decomposition.

Pulling the inside out, it could be useful to define entire linear algebra objects as single symbols.
The reason this is not done in *saltype* is the *LT* in the name,
and the horrible number (and variants) of binary operators to consider.

Conditionals
------------
The :py:obj:`select() <saltype.datatype.Salt.select>` method is a primitive
conditional, but for the sake of differentiability, such support is on purpose
kept to a minimum. In the end, conditionals are not differentiable, and the
approach in *saltype* is just pragmatic: *Nobody is going to hit that corner.*

Stack-size
----------
The initially presented example::

    from saltype import Leaf, sin
    y = x = Leaf(3.14159)
    for _ in range(100):
        y = sin(y) * y

is nice, but what happens if you increase the *range* argument to 1000?
Most likely, there will be some error messages about maximum recursion depth.
For most actual applications, this should not pose any problem. Hence if it
happens, consider first whether the way your implementation works is as
intended.

If really necessary, do this::

    from sys import setrecursionlimit
    setrecursionlimit(2000) # or whatever you need
