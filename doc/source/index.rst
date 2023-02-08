SALT Documentation
==================
Brief description
-----------------
**SALT** stands for **S**\ ymbolic **A**\ lgebra **L**\ igh\ **T**. It's
main purpose is to provide symbolic derivatives for systems of several
thousand variables efficiently in terms of performance and memory. With that,
the objective is not to be fit to hold a candle to established packages like
sympy_ or CaSAdi_, but to have a lightweight package to fulfil particular
needs. The main feature and strategy is that performance of core functionality
is not compromised by any additional functionality or convenience.

In its current version, it is even pure python. Porting into a *C*-extension
is possible for further performance gain, but the subsequent
compiler/version/platform dependencies freak me out.

:Date:
    |date|
:Author:
    Volker Siepmann <volker.siepmann@gmail.com>

.. |date| date:: %d.%m.%Y

Contents:

.. toctree::
    :maxdepth: 2

    introduction
    api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
